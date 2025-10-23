#!/usr/bin/env python3
"""
Rescore with probabilistic OR logic across future hours and neighbor cells.
Used to check motion-aware or "any future hit" scoring from stored probabilities.
"""

import numpy as np
import pandas as pd
import argparse
import gzip, os
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# -----------------------------------------------------------------------------
# Utility: infer grid shape safely
# -----------------------------------------------------------------------------
def infer_grid_shape(df: pd.DataFrame):
    """Infer (H, W, T) safely from first hour slice."""
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df.sort_values(["time", "lat", "lon"], inplace=True, ignore_index=True)
    hours = df["time"].dt.floor("h")
    first = hours.iloc[0]
    g0 = df.loc[hours == first, ["lat", "lon"]]
    H = g0["lat"].nunique()
    W = g0["lon"].nunique()
    T = df["time"].dt.floor("h").nunique()
    print(f"[shape] H={H}  W={W}  T={T}  len(df)={len(df)}")
    return H, W, T


# -----------------------------------------------------------------------------
# Probabilistic OR scoring (over space/time)
# -----------------------------------------------------------------------------
def prob_or_time(df, p, lead_h):
    """Compute probabilistic OR of future probabilities within lead_h for each point."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df.sort_values(["time", "lat", "lon"], inplace=True, ignore_index=True)
    df["p"] = p

    out = np.zeros_like(p)
    dt = np.timedelta64(int(lead_h), "h")

    grouped = df.groupby(["lat", "lon"], sort=False, group_keys=False)
    for i, g in enumerate(grouped):
        g = g[1].sort_values("time")
        t = g["time"].to_numpy()
        prob = g["p"].to_numpy()
        out_idx = g.index.to_numpy()
        new_vals = np.zeros_like(prob)
        for j in range(len(prob)):
            mask = (t >= t[j]) & (t <= t[j] + dt)
            new_vals[j] = 1 - np.prod(1 - prob[mask])
        out[out_idx] = new_vals
        if i % 1000 == 0:
            print(f"  processed {i} / {len(grouped)} lat-lon cells", flush=True)
    return out


# -----------------------------------------------------------------------------
# Future max label
# -----------------------------------------------------------------------------
def future_max_label_by_point(df, target_col, hours):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df.sort_values(["time", "lat", "lon"], inplace=True, ignore_index=True)
    dt = np.timedelta64(int(hours), "h")

    out = np.zeros(len(df), dtype=int)
    for (lat, lon), g in df.groupby(["lat", "lon"], sort=False):
        g = g.sort_values("time")
        t = g["time"].to_numpy()
        vals = g[target_col].to_numpy()
        new = np.zeros_like(vals)
        for i in range(len(vals)):
            mask = (t >= t[i]) & (t <= t[i] + dt)
            new[i] = np.max(vals[mask])
        out[g.index] = new
    return out


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def metrics(y_true, y_score):
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = np.nan
    try:
        pr = average_precision_score(y_true, y_score)
    except ValueError:
        pr = np.nan
    try:
        brier = brier_score_loss(y_true, y_score)
    except ValueError:
        brier = np.nan
    return auc, pr, brier


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Rescore probabilistic OR across future hours and neighbors.")
    parser.add_argument("--labelled", required=True, help="Labelled grid CSV.gz file.")
    parser.add_argument("--probs-npy", required=True, help="Path to saved probs.npy (from lead_eval).")
    parser.add_argument("--target", required=True, choices=["storm", "near_storm", "pregen"], help="Target column.")
    parser.add_argument("--lead-hours", nargs="+", type=int, required=True, help="Lead hours to evaluate.")
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument("--advect-cells", type=int, default=1)
    parser.add_argument("--grid-height", type=int, default=None)
    parser.add_argument("--grid-width", type=int, default=None)
    args = parser.parse_args()

    print("[PROB-OR | COINCIDENT]", flush=True)
    df = pd.read_csv(args.labelled, compression="infer", low_memory=False)
    df[args.target] = df[args.target].fillna(0).astype(int)

    p = np.load(args.probs_npy)
    assert len(p) == len(df), f"Mismatch: probs({len(p)}) vs df({len(df)})"

    H, W, T = infer_grid_shape(df)

    auc, pr, brier = metrics(df[args.target].to_numpy(), p)
    print(f"COINCIDENT   →  AUC={auc:.3f}  PRAUC={pr:.3f}  Brier={brier:.3f}  Pos={df[args.target].sum():,}/{len(df):,}")

    for h in args.lead_hours:
        print(f"Preparing lead +{h}h probs & labels …", flush=True)
        p_any = prob_or_time(df[["time","lat","lon"]], p, h)
        y = future_max_label_by_point(df[["time","lat","lon",args.target]].copy(), args.target, h)
        auc, pr, brier = metrics(y, p_any)
        print(f"Lead +{h}h    →  AUC={auc:.3f}  PRAUC={pr:.3f}  Brier={brier:.3f}  Pos={y.sum():,}/{len(y):,}", flush=True)


if __name__ == "__main__":
    main()