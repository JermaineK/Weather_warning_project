#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_runner.py

Grid alert sweep over thresholds / persistence / neighborhood / hourly-quantile.

- Supports modern per-lead model bundle with keys:
    {
        "features": [...],
        "imputer_stats": {...},
        "clip_stats": {"lo": [...], "hi": [...]},
        "scaler": StandardScaler or None,
        "meta": {...},
        "models": {
            "global": sklearn_estimator,   # optional
            "24": sklearn_estimator,      # per-lead models keyed by lead hours (str or int)
            "48": ...,
            ...
        }
    }

- Also supports legacy bundles:
    {"model","scaler","features"} or plain sklearn estimator objects.

- Reads labelled grid (CSV(.gz)/Parquet) and produces a sweep summary with:

    lead, thr, persist, neighbors, quantile,
    AUC, PRAUC, Brier,
    F1, Precision, Recall,
    Coverage, alerts, pos, rows

Target labels are built with a time-aware future window per (lat,lon):

    y_L(t,cell) = 1 if any base_label==1 occurs in (t, t+L] for that cell.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


# ---------------- I/O helpers ----------------


def read_any(
    path: Path | str,
    parse_dates: Optional[List[str]] = None,
    usecols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Read CSV(.gz) or Parquet with optional column subset.

    - For Parquet: uses `columns=usecols` if provided.
    - For CSV: uses `usecols` filter + memory_map to ease RAM pressure.
    """
    p = str(path).lower()
    if p.endswith((".parquet", ".pq", ".pqt")):
        return pd.read_parquet(path, columns=usecols if usecols else None)

    return pd.read_csv(
        path,
        low_memory=False,
        compression="infer",
        parse_dates=parse_dates if parse_dates else None,
        usecols=usecols if usecols else None,
        memory_map=True,
    )


def write_any(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    p = str(path).lower()
    if p.endswith((".parquet", ".pq", ".pqt")):
        df.to_parquet(path, index=False)
    else:
        comp = "gzip" if p.endswith(".gz") else "infer"
        df.to_csv(path, index=False, compression=comp)


# ---------------- model / bundle helpers ----------------


def load_model_bundle(
    model_path: str | Path,
) -> Tuple[Any, Dict[int, Any], Optional[Any], List[str], Optional[Dict[str, float]], Optional[Dict[str, np.ndarray]]]:
    """
    Load flexible model bundle formats.

    Returns:
        global_model
        per_lead_models: {lead_h: estimator}
        scaler
        features
        imputer_stats
        clip_stats
    """
    m = joblib.load(model_path)

    global_model = None
    per_lead: Dict[int, Any] = {}
    scaler = None
    features: List[str] = []
    imputer_stats: Optional[Dict[str, float]] = None
    clip_stats: Optional[Dict[str, np.ndarray]] = None

    # Dict-style bundle (newer format)
    if isinstance(m, dict):
        # Common keys
        features = list(m.get("features") or m.get("feats") or [])
        scaler = m.get("scaler", None)
        imputer_stats = m.get("imputer_stats", None)
        clip_stats = m.get("clip_stats", None)

        # Try explicit per-lead container first
        if "per_lead_models" in m and isinstance(m["per_lead_models"], dict):
            for k, v in m["per_lead_models"].items():
                try:
                    lead = int(k)
                    per_lead[lead] = v
                except Exception:
                    # Non-numeric keys here are ignored; global may still be in 'model'
                    pass

        # Fallback: generic "models" dict that may contain "global" and per-lead keys
        elif "models" in m and isinstance(m["models"], dict):
            for k, v in m["models"].items():
                try:
                    lead = int(k)
                    per_lead[lead] = v
                except Exception:
                    # Non-numeric: treat as global default model
                    global_model = v

        # Global model fallbacks
        if global_model is None:
            global_model = (
                m.get("model")
                or m.get("model-out")
                or m.get("estimator")
                or m.get("pipe")
            )

        if global_model is None and not per_lead:
            raise ValueError(
                "Unsupported model bundle format: expected 'model', 'models', or 'per_lead_models' keys."
            )

        if not features:
            # Some bundles may stash feature names on meta or model
            meta_feats = []
            meta = m.get("meta", {})
            if isinstance(meta, dict):
                meta_feats = list(meta.get("features", []))
            if meta_feats:
                features = meta_feats
            else:
                raise ValueError("Model bundle lacks 'features' list; cannot build X matrix.")

        return global_model, per_lead, scaler, features, imputer_stats, clip_stats

    # Plain estimator
    global_model = m
    scaler = getattr(m, "scaler_", None)
    feat_arr = getattr(m, "feature_names_in_", None) or getattr(m, "features_", None)
    if feat_arr is None:
        raise ValueError(
            "Plain estimator bundle has no 'feature_names_in_' or 'features_'; "
            "please save as a dict bundle with 'features'."
        )
    features = list(feat_arr)
    return global_model, per_lead, scaler, features, None, None


def build_feature_matrix(
    df: pd.DataFrame,
    features: List[str],
    imputer_stats: Optional[Dict[str, float]],
    clip_stats: Optional[Dict[str, np.ndarray]],
    scaler: Optional[Any],
) -> np.ndarray:
    """
    Align columns to features; impute, optional clip, then scale.
    """
    Xdf = pd.DataFrame(index=df.index)
    for c in features:
        if c in df.columns:
            Xdf[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            Xdf[c] = np.nan

    X = Xdf.to_numpy(dtype=float, copy=True)
    X[~np.isfinite(X)] = np.nan

    # Impute
    if imputer_stats:
        for j, c in enumerate(features):
            fill = float(imputer_stats.get(c, 0.0))
            m = ~np.isfinite(X[:, j])
            if m.any():
                X[m, j] = fill
    else:
        X = np.where(np.isfinite(X), X, 0.0)

    # Clip
    if clip_stats and "lo" in clip_stats and "hi" in clip_stats:
        lo = np.asarray(clip_stats["lo"])
        hi = np.asarray(clip_stats["hi"])
        if lo.shape == X.shape[1:] and hi.shape == X.shape[1:]:
            X = np.clip(X, lo, hi)

    # Scale
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            # If transform fails, fall back to unscaled
            pass

    return X


def score_with_estimator(est: Any, X: np.ndarray) -> np.ndarray:
    """
    Get probabilities p(y=1|x) from an estimator.
    """
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1].astype(np.float64)
    if hasattr(est, "decision_function"):
        from scipy.special import expit

        dec = est.decision_function(X).astype(np.float64)
        return expit(dec)
    # last-ditch: try predict as probabilities-ish (0/1)
    pred = est.predict(X).astype(np.float64)
    return np.clip(pred, 0.0, 1.0)


# ---------------- utilities ----------------


def parse_float_list(val: str) -> List[float]:
    # "a:b:c" → a, a+c, ..., ≤b; or "x,y,z"
    s = str(val).strip()
    if ":" in s:
        a, b, c = s.split(":")
        a, b, c = float(a), float(b), float(c)
        n = int(math.floor((b - a) / c + 1e-9)) + 1
        return [round(a + i * c, 10) for i in range(n)]
    return [float(x) for x in s.split(",") if str(x).strip()]


def future_max_timeaware(df: pd.DataFrame, target: str, hours: int) -> np.ndarray:
    """
    For each row (lat,lon,t), returns 1 if any target==1 occurs in (t, t+hours] for that cell.
    Handles missing hours and irregular cadence via timestamp searchsorted.
    """
    out = np.zeros(len(df), dtype=np.int8)
    hour_ns = np.int64(hours) * np.int64(3_600_000_000_000)

    for (_, _), g in df.groupby(["lat", "lon"], sort=False, group_keys=False):
        idx = g.index.to_numpy()
        y = pd.to_numeric(g[target], errors="coerce").fillna(0).astype(np.int8).to_numpy()
        t_ns = pd.to_datetime(g["time"], errors="coerce", utc=True).view("int64").to_numpy()

        order = np.argsort(t_ns, kind="mergesort")
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))

        t_sorted = t_ns[order]
        y_sorted = y[order]

        ps = np.zeros(len(y_sorted) + 1, dtype=np.int64)
        ps[1:] = np.cumsum(y_sorted)

        t_end_sorted = t_sorted + hour_ns
        end_pos = np.searchsorted(t_sorted, t_end_sorted, side="right")

        any_future_sorted = (ps[end_pos] - ps[np.arange(len(y_sorted)) + 1]) > 0
        any_future = any_future_sorted[inv].astype(np.int8)
        out[idx] = any_future

    return out


def apply_persistence(df: pd.DataFrame, base_mask: np.ndarray, persist_h: int) -> np.ndarray:
    """Require ≥ persist_h consecutive kept hours within each (lat,lon)."""
    if persist_h <= 1:
        return base_mask
    out = np.zeros(len(df), dtype=bool)
    tmp = df.assign(_flag=base_mask.astype(np.int8))

    for (_, _), g in tmp.groupby(["lat", "lon"], sort=False, group_keys=False):
        f = g["_flag"].to_numpy()
        if f.sum() == 0:
            continue
        keep = (
            pd.Series(f)
            .rolling(persist_h, min_periods=persist_h)
            .sum()
            .to_numpy()
            >= persist_h
        )
        out[g.index.to_numpy()] = keep
    return out


def indexize(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    ilat = pd.factorize(df["lat"].round(6))[0]
    ilon = pd.factorize(df["lon"].round(6))[0]
    return ilat, ilon


def neighborhood_filter(df: pd.DataFrame, flag_mask: np.ndarray, min_neighbors: int) -> np.ndarray:
    """Keep cells that have ≥ min_neighbors of 8-neighbors flagged at the same hour."""
    if min_neighbors <= 0:
        return flag_mask
    kept = np.zeros(len(df), dtype=bool)
    ilat, ilon = indexize(df)
    meta = pd.DataFrame(
        {
            "time": df["time"],
            "ilat": ilat,
            "ilon": ilon,
            "flag": flag_mask.astype(np.int8),
        }
    )

    for _, g in meta.groupby("time", sort=False):
        if g["flag"].sum() == 0:
            continue
        flagged = g.loc[g["flag"] == 1, ["ilat", "ilon"]].to_numpy()
        present = set((int(i), int(j)) for i, j in flagged)
        nbr_ct = np.zeros(len(g), dtype=np.int16)
        IL = g["ilat"].to_numpy()
        JL = g["ilon"].to_numpy()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                nbr_ct += np.fromiter(
                    ((IL[k] + di, JL[k] + dj) in present for k in range(len(g))),
                    count=len(g),
                    dtype=np.int16,
                )
        keep_h = (nbr_ct >= int(min_neighbors)) & (g["flag"].to_numpy().astype(bool))
        kept[g.index.to_numpy()] = keep_h
    return kept


def hourly_throttle(
    df: pd.DataFrame,
    score: np.ndarray,
    mask_in: np.ndarray,
    keep_quantile: Optional[float],
) -> np.ndarray:
    """Within each hour, among candidates mask_in, keep those with score ≥ quantile."""
    if keep_quantile is None:
        return mask_in
    out = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby("time", sort=False).indices.items():
        sub = mask_in[idx]
        if not sub.any():
            continue
        s = score[idx][sub]
        q = np.quantile(s, keep_quantile) if s.size > 1 else s[0]
        keep = np.zeros_like(sub)
        keep[sub] = score[idx][sub] >= q
        out[idx] = keep
    return out


# ---------------- main ----------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid alert sweep runner (per-lead model aware).")
    ap.add_argument("--labelled", required=True, help="Labelled grid CSV(.gz) or Parquet.")
    ap.add_argument(
        "--model",
        required=True,
        help="Model bundle: dict with 'features' + ('models' or 'per_lead_models'), "
        "or legacy {'model','scaler','features'}, or plain estimator.",
    )
    ap.add_argument(
        "--target",
        default="pregen",
        choices=["storm", "near_storm", "pregen"],
        help="Base label to roll into future time windows.",
    )
    ap.add_argument(
        "--leads",
        default="24",
        help="Comma list of lead hours, e.g. '24,48,72,120'.",
    )
    ap.add_argument(
        "--thr-grid",
        default="0.04:0.10:0.002",
        help="Threshold grid: 'a:b:c' or comma list (e.g. 0.02,0.05,0.08).",
    )
    ap.add_argument(
        "--persist",
        default="2",
        help="Comma list of consecutive-hour minima (e.g. '1,2').",
    )
    ap.add_argument(
        "--neighbors",
        default="3",
        help="Comma list of min 8-neighbors (0 disables neighborhood filter).",
    )
    ap.add_argument(
        "--quantiles",
        default="0.90",
        help="Comma list of hourly keep quantiles (e.g. '0.90' keeps top 10%% per hour).",
    )
    ap.add_argument(
        "--subsample-hours",
        type=str,
        default="1.0",
        help="Fraction (0<q≤1) or stride like 'every:3' to keep each n-th hour.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for hourly subsampling when using a fraction.",
    )
    ap.add_argument(
        "--out",
        default="results/sweep_summary.csv",
        help="Output CSV/Parquet with sweep metrics.",
    )

    args = ap.parse_args()

    leads = [int(x) for x in str(args.leads).split(",") if str(x).strip()]
    thr_grid = parse_float_list(args.thr_grid)
    persists = [int(x) for x in str(args.persist).split(",") if str(x).strip()]
    neighbors = [int(x) for x in str(args.neighbors).split(",") if str(x).strip()]
    quantiles = [float(x) for x in str(args.quantiles).split(",") if str(x).strip()]

    print("== Sweep runner ==")
    print("Labelled :", args.labelled)
    print("Model    :", args.model)
    print("Target   :", args.target)
    print("Leads    :", leads)
    if len(thr_grid) > 6:
        print("Thresh   :", thr_grid[:3], "…", thr_grid[-3:])
    else:
        print("Thresh   :", thr_grid)
    print("Persist  :", persists, "Neighbors:", neighbors, "Quantiles:", quantiles)
    print("Subsample:", args.subsample_hours)
    print("Seed     :", args.seed)

    # Load model bundle
    global_model, per_lead_models, scaler, feat_list, imp_stats, clip_stats = load_model_bundle(
        args.model
    )

    labelled_path = Path(args.labelled)

    # Figure out minimal set of columns we actually need
    need_cols = set(feat_list) | {"time", "lat", "lon", args.target}

    # Load labelled data with *only* required columns to reduce memory
    df = read_any(labelled_path, parse_dates=["time"], usecols=list(need_cols))

    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Labelled file missing required columns: {missing}")

    # Core typing & sorting
    df = df.loc[:, list(need_cols)].copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True).dt.tz_localize(
        None
    )
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["time", "lat", "lon"]).sort_values(
        ["time", "lat", "lon"], kind="mergesort"
    )
    df = df.reset_index(drop=True)

    # Optional hour subsampling
    sub = str(args.subsample_hours).strip().lower()
    if sub.startswith("every:"):
        try:
            step = int(sub.split(":", 1)[1])
            hours = df["time"].drop_duplicates().sort_values()
            keep_hours = set(hours.iloc[:: max(1, step)].tolist())
            df = df[df["time"].isin(keep_hours)].copy()
        except Exception:
            pass
    else:
        frac = float(sub)
        if 0 < frac < 1.0:
            rng = np.random.RandomState(args.seed)
            hours = df["time"].drop_duplicates().sort_values()
            keep_n = max(1, int(round(len(hours) * frac)))
            keep_idx = rng.choice(len(hours), size=keep_n, replace=False)
            keep_hours = set(hours.iloc[np.sort(keep_idx)].tolist())
            df = df[df["time"].isin(keep_hours)].copy()

    df = df.sort_values(["time", "lat", "lon"], kind="mergesort").reset_index(drop=True)
    print(f"Rows after subsample: {len(df):,}")

    # Ensure base target is 0/1
    y_base = pd.to_numeric(df[args.target], errors="coerce").fillna(0)
    df[args.target] = (y_base > 0).astype(np.int8)

    # Build feature matrix once (aligned to bundle)
    X = build_feature_matrix(df, feat_list, imputer_stats=imp_stats, clip_stats=clip_stats, scaler=scaler)

    rows_out: List[Dict[str, Any]] = []

    for lead in leads:
        # Pick the estimator for this lead (per-lead model if available, else global)
        est = per_lead_models.get(lead, global_model)
        if lead in per_lead_models:
            print(f"[lead {lead}] using per-lead estimator from bundle")
        else:
            print(f"[lead {lead}] using global estimator from bundle")

        # Score probabilities
        p = score_with_estimator(est, X)

        # Build time-aware labels for this lead
        y = future_max_timeaware(df, args.target, lead)
        pos_total = int(y.sum())

        # Baseline metrics
        if pos_total == 0 or pos_total == len(y):
            auc = np.nan
            prauc = np.nan
        else:
            auc = roc_auc_score(y, p)
            prauc = average_precision_score(y, p)
        brier = brier_score_loss(y, p)

        print(
            f"[lead {lead}] positives={pos_total:,}/{len(y):,} "
            f"(frac={pos_total/len(y):.4f})  AUC={auc if not np.isnan(auc) else float('nan'):.3f} "
            f"PRAUC={prauc if not np.isnan(prauc) else float('nan'):.3f}  Brier={brier:.4f}"
        )

        for thr in thr_grid:
            base_mask = p >= thr
            for ph in persists:
                keep_persist = apply_persistence(df, base_mask, ph)
                for nb in neighbors:
                    keep_nb = neighborhood_filter(df, keep_persist, nb)
                    for q in quantiles:
                        keep_final = hourly_throttle(df, p, keep_nb, q)

                        pred_pos = int(keep_final.sum())
                        if pred_pos == 0:
                            rows_out.append(
                                dict(
                                    lead=lead,
                                    thr=thr,
                                    persist=ph,
                                    neighbors=nb,
                                    quantile=q,
                                    AUC=float(auc) if not np.isnan(auc) else np.nan,
                                    PRAUC=float(prauc) if not np.isnan(prauc) else np.nan,
                                    Brier=float(brier),
                                    F1=np.nan,
                                    Precision=np.nan,
                                    Recall=np.nan,
                                    Coverage=0.0,
                                    alerts=0,
                                    pos=pos_total,
                                    rows=len(df),
                                )
                            )
                            continue

                        tp = int((keep_final & (y == 1)).sum())
                        fp = pred_pos - tp
                        fn = pos_total - tp

                        precision = tp / pred_pos if pred_pos > 0 else np.nan
                        recall = tp / pos_total if pos_total > 0 else np.nan
                        if (precision + recall) > 0:
                            f1 = 2 * precision * recall / (precision + recall)
                        else:
                            f1 = np.nan
                        coverage = pred_pos / float(len(df))

                        rows_out.append(
                            dict(
                                lead=lead,
                                thr=thr,
                                persist=ph,
                                neighbors=nb,
                                quantile=q,
                                AUC=float(auc) if not np.isnan(auc) else np.nan,
                                PRAUC=float(prauc) if not np.isnan(prauc) else np.nan,
                                Brier=float(brier),
                                F1=float(f1) if not np.isnan(f1) else np.nan,
                                Precision=float(precision) if not np.isnan(precision) else np.nan,
                                Recall=float(recall) if not np.isnan(recall) else np.nan,
                                Coverage=float(coverage),
                                alerts=pred_pos,
                                pos=pos_total,
                                rows=len(df),
                            )
                        )

    out_path = Path(args.out)
    write_any(out_path, pd.DataFrame(rows_out))
    print(f"Wrote {out_path} rows: {len(rows_out)}")


if __name__ == "__main__":
    main()