#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
best_f1.py — Find best-F1 probability thresholds per lead using time-aware labels.

Supported model formats
-----------------------
A) Per-lead bundle:
    {
      "per_lead_models": { 24: sklearn_pipeline, 48: pipeline, ... },
      "features": [...]
    }

B) Single-model bundle:
    {
      "model": sklearn_estimator,
      "scaler": sklearn_scaler_or_None,
      "features": [...]
    }

Time-aware future label
-----------------------
For lead L hours:
    y_L(t, cell) = 1 if any base_label==1 occurs in (t, t+L] for that (lat,lon).

Output
------
Prints best F1 threshold per lead; optionally saves a CSV.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_recall_curve


# ------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------

def read_any(path: str | Path,
             usecols: Iterable[str] | None = None,
             parse_dates: Iterable[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    p = path.suffix.lower()
    if p in {".parquet", ".pq", ".pqt"}:
        return pd.read_parquet(path, columns=usecols if usecols else None)
    return pd.read_csv(
        path,
        usecols=usecols if usecols else None,
        parse_dates=list(parse_dates) if parse_dates else None,
        low_memory=False,
        compression="infer",
    )


# ------------------------------------------------------------
# Model loaders
# ------------------------------------------------------------

def load_bundle(bundle_path: str | Path):
    """Return (model_type, info_dict)."""
    m = joblib.load(bundle_path)
    if isinstance(m, dict) and "per_lead_models" in m:
        return "per_lead", {
            "per_lead_models": m["per_lead_models"],
            "features": list(m.get("features", []))
        }
    if isinstance(m, dict) and all(k in m for k in ("model", "scaler", "features")):
        return "single", {
            "model": m["model"],
            "scaler": m["scaler"],
            "features": list(m["features"]),
        }
    raise ValueError(
        "Model bundle must contain either:\n"
        "  {'per_lead_models', 'features'}  OR  {'model','scaler','features'}"
    )


# ------------------------------------------------------------
# Time-aware future label
# ------------------------------------------------------------

def future_max_timeaware(group: pd.DataFrame, label_col: str, hours: int) -> np.ndarray:
    """
    Strict-future window: y_L = 1 if any label==1 in (t, t+L].
    Returned vector aligns with group.index.
    """
    y = pd.to_numeric(group[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
    t = pd.to_datetime(group["time"], utc=True, errors="coerce")
    t_ns = t.view("int64").to_numpy()

    order = np.argsort(t_ns, kind="mergesort")
    inv   = np.empty_like(order)
    inv[order] = np.arange(len(order))

    t_sorted = t_ns[order]
    y_sorted = y[order]

    ps = np.zeros(len(y_sorted) + 1, dtype=np.int64)
    ps[1:] = np.cumsum(y_sorted)

    h_ns = np.int64(hours) * np.int64(3_600_000_000_000)
    t_end = t_sorted + h_ns
    end_pos = np.searchsorted(t_sorted, t_end, side="right")

    # Use (i+1 .. end_pos-1) -> implements strict future (no current hour).
    any_future_sorted = (ps[end_pos] - ps[np.arange(len(y_sorted)) + 1]) > 0
    return any_future_sorted[inv].astype(np.int8)


# ------------------------------------------------------------
# Best-F1 from precision-recall curve
# ------------------------------------------------------------

def best_f1_from_scores(y_true: np.ndarray, p: np.ndarray) -> dict:
    pr, rc, th = precision_recall_curve(y_true, p)
    if len(pr) == 0:
        return dict(best_f1=0.0, threshold=0.5, precision=0.0, recall=0.0)

    f1 = 2 * pr * rc / (pr + rc + 1e-9)
    i = int(np.nanargmax(f1))

    # sklearn returns len(th)==len(pr)-1
    thr = float(th[max(i - 1, 0)]) if len(th) else 0.5

    return dict(
        best_f1=float(f1[i]),
        threshold=thr,
        precision=float(pr[i]),
        recall=float(rc[i]),
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Best-F1 thresholds per lead (time-aware).")
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--leads", nargs="+", type=int, default=[24])
    ap.add_argument("--target",
                    choices=["pregen", "storm", "near_storm", "storm_hit"],
                    default="pregen")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-csv", default=None)
    args = ap.parse_args()

    # -------- Load model bundle --------
    mtype, info = load_bundle(args.model)

    if mtype == "per_lead":
        per_lead_models = info["per_lead_models"]
        features = info["features"]
        print(f"[model] Per-lead bundle with {len(per_lead_models)} models.")
    else:
        model = info["model"]
        scaler = info["scaler"]
        features = info["features"]
        print("[model] Single model bundle")

    # -------- Load labelled grid --------
    need_cols = {"time", "lat", "lon", args.target, *features}
    df = read_any(args.labelled, parse_dates=["time"])
    df = df[list(col for col in df.columns if col in need_cols)].copy()

    if args.limit:
        df = df.iloc[:args.limit].copy()

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["time", "lat", "lon"]).reset_index(drop=True)
    df = df.sort_values(["lat", "lon", "time"], kind="mergesort").reset_index(drop=True)

    df[args.target] = (pd.to_numeric(df[args.target], errors="coerce").fillna(0) > 0).astype(int)

    # -------- Single model: pre-compute scores --------
    if mtype == "single":
        X = df[features].to_numpy(float)
        if scaler is not None:
            X = scaler.transform(X)
        p_all = model.predict_proba(X)[:, 1]

    # -------- Compute best F1 per lead --------
    out_rows = []

    for h in args.leads:
        print(f"\n[lead +{h}h] computing future-window labels…")

        # Build y_L
        yL = (
            df.groupby(["lat", "lon"], sort=False, group_keys=False)
              .apply(lambda g: pd.Series(
                  future_max_timeaware(g, args.target, hours=h),
                  index=g.index))
              .sort_index()
              .to_numpy()
              .astype(int)
        )

        # Choose model
        if mtype == "per_lead":
            if h not in per_lead_models:
                print(f"  WARNING: no model for lead {h}h -> skipping.")
                continue
            pipe = per_lead_models[h]
            X = df[features].to_numpy(float)
            p = pipe.predict_proba(X)[:, 1]
        else:
            p = p_all

        res = best_f1_from_scores(yL, p)
        out_rows.append({"lead_h": h, **res})

        print(
            f"  Best F1={res['best_f1']:.3f} @ thr={res['threshold']:.3f}  "
            f"(P={res['precision']:.3f}, R={res['recall']:.3f})  "
            f"Positives={int(yL.sum())}/{len(yL)}"
        )

    # -------- Save output CSV --------
    if args.save_csv and out_rows:
        out_df = pd.DataFrame(out_rows).sort_values("lead_h")
        Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.save_csv, index=False)
        print(f"\nSaved -> {args.save_csv}")


if __name__ == "__main__":
    main()