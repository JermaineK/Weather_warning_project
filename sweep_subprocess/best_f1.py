#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
best_f1.py
Find best-F1 thresholds per lead using either:
  (A) single model bundle: {"model","scaler","features"}
  (B) per-lead bundle: {"per_lead_models": {lead_h: sklearn Pipeline, ...}, "features": [...]}

Time-aware future window labels:
  y_L(t,cell) = 1 if any base_label==1 occurs in (t, t+L] within that lat/lon group.

Examples
--------
# Per-lead bundle
python best_f1.py --labelled data/grid_labelled_FMA_gka.csv.gz \
                  --model models/grid_logit.pkl \
                  --leads 24 48 72 120 \
                  --target storm \
                  --save-csv results/best_f1_thresholds.csv

# Single model (legacy)
python best_f1.py --labelled data/grid_labelled_base.csv.gz \
                  --model models/logit_labelled.pkl \
                  --leads 24 48 72 \
                  --target pregen
"""

import argparse, numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.metrics import precision_recall_curve


def read_any(path, usecols=None, parse_dates=None):
    p = str(path).lower()
    if p.endswith((".parquet",".pq",".pqt")):
        return pd.read_parquet(path, columns=usecols if usecols else None)
    return pd.read_csv(path, usecols=usecols if usecols else None,
                       parse_dates=parse_dates if parse_dates else None,
                       low_memory=False)


def best_f1_from_scores(y_true: np.ndarray, p: np.ndarray):
    pr, rc, th = precision_recall_curve(y_true, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-9)
    i = int(np.nanargmax(f1)) if len(f1) else 0
    # sklearn's `th` has length len(pr)-1
    thr = float(th[max(i - 1, 0)]) if len(th) else 0.5
    return dict(best_f1=float(f1[i] if len(f1) else 0.0),
                threshold=thr,
                precision=float(pr[i] if len(pr) else 0.0),
                recall=float(rc[i] if len(rc) else 0.0))


def future_max_timeaware(group_df: pd.DataFrame, label_col: str, hours: int) -> np.ndarray:
    """
    Time-aware future window per (lat,lon) group.
    Returns an array aligned to group_df.index.
    """
    y = pd.to_numeric(group_df[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
    t = pd.to_datetime(group_df["time"], errors="coerce", utc=True)
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

    # Any future positives strictly after current index -> use (i+1 .. end_pos-1)
    any_future_sorted = (ps[end_pos] - ps[np.arange(len(y_sorted)) + 1]) > 0
    any_future = any_future_sorted[inv].astype(np.int8)
    return any_future


def main():
    ap = argparse.ArgumentParser(description="Best-F1 thresholds per lead with time-aware future windows.")
    ap.add_argument("--labelled", required=True, help="Path to labelled grid CSV/Parquet")
    ap.add_argument("--model", required=True, help="Path to joblib bundle (per-lead or single)")
    ap.add_argument("--leads", nargs="+", type=int, default=[24], help="Lead hours (e.g., 6 12 24 48)")
    ap.add_argument("--target", default="pregen",
                    choices=["pregen","storm","near_storm"], help="Base label to roll into the future window")
    ap.add_argument("--limit", type=int, default=None, help="Optional row cap for quick runs")
    ap.add_argument("--save-csv", default=None, help="Optional output CSV of thresholds")
    args = ap.parse_args()

    bundle = joblib.load(args.model)

    # Identify bundle type
    per_lead = isinstance(bundle, dict) and "per_lead_models" in bundle
    if per_lead:
        # Expect sklearn Pipelines in per_lead_models; they include scaler internally
        per_lead_models = bundle["per_lead_models"]
        features = list(bundle.get("features", []))
        print(f"[model] Detected per-lead bundle with {len(per_lead_models)} models. "
              f"Using features: {len(features)}")
    else:
        # Legacy single model + scaler + features
        if not all(k in bundle for k in ("model","scaler","features")):
            raise ValueError("Model bundle must contain either per_lead_models or (model, scaler, features).")
        model = bundle["model"]
        scaler = bundle["scaler"]
        features = list(bundle["features"])
        print(f"[model] Detected single model. Using features: {len(features)}")

    need_cols = {"time","lat","lon", args.target, *features}
    df = read_any(args.labelled, usecols=lambda c: c in need_cols, parse_dates=["time"])
    if args.limit:
        df = df.iloc[:args.limit].copy()

    # Type & sorting hygiene
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True).dt.tz_localize(None)
    df["lat"]  = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"]  = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["time","lat","lon"]).sort_values(["lat","lon","time"], kind="mergesort").reset_index(drop=True)

    # Coerce base target to 0/1
    y_base = pd.to_numeric(df[args.target], errors="coerce").fillna(0)
    df[args.target] = (y_base > 0).astype(int)

    out_rows = []

    # Score once for single-model bundles; for per-lead, score per lead with its own pipeline
    if not per_lead:
        X = scaler.transform(df[features].to_numpy(float))
        p_all = model.predict_proba(X)[:, 1]

    # Compute best F1 per requested lead
    for h in args.leads:
        # Build time-aware future window labels per cell
        yL = (
            df.groupby(["lat","lon"], sort=False, group_keys=False)
              .apply(lambda g: pd.Series(future_max_timeaware(g, args.target, hours=h), index=g.index))
              .sort_index()
              .to_numpy()
              .astype(int)
        )

        # Score: choose the right model
        if per_lead:
            if int(h) not in per_lead_models:
                print(f"[warn] lead {h}h not found in per-lead bundle; skipping.")
                continue
            pipe = per_lead_models[int(h)]
            p = pipe.predict_proba(df[features].to_numpy(float))[:, 1]
        else:
            p = p_all

        res = best_f1_from_scores(yL, p)
        out_rows.append({"lead_h": int(h), **res})
        print(f"Lead +{int(h):>3}h → Best F1: {res['best_f1']:.3f} @ thr {res['threshold']:.3f} "
              f"(P={res['precision']:.3f}, R={res['recall']:.3f}) "
              f"Pos={int(yL.sum())}/{len(yL)}")

    if args.save_csv and out_rows:
        out_df = pd.DataFrame(out_rows).sort_values("lead_h")
        Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.save_csv, index=False)
        print(f"Saved thresholds → {args.save_csv}")


if __name__ == "__main__":
    main()