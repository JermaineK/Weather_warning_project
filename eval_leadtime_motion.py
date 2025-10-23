#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lead-time evaluation with motion-aware labels (neighbor dilation + optional advective padding),
progress bars, chunked probability scoring, and non-fatal hooks for rainfall/lightning.

Example (PowerShell):
  python .\eval_leadtime_motion.py `
    --labelled "data\grid_labelled_FMA_gka.csv.gz" `
    --model "models\grid_logit_pregen_gka_cal.pkl" `
    --target pregen `
    --lead-hours 24 48 `
    --neighbor-radius 1 `
    --advect-cells 1 `
    --chunk-rows 2000000 `
    --checkpoint-dir "results\lead_eval_ckpt_motion"
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# -----------------------------
# Small helpers
# -----------------------------
def println(msg): print(msg, flush=True)

def load_model(model_path):
    M = joblib.load(model_path)
    if isinstance(M, dict) and all(k in M for k in ("model","scaler","features")):
        return M["model"], M["scaler"], list(M["features"])
    raise ValueError("Model file must be a dict with keys: 'model', 'scaler', 'features'.")

# -----------------------------
# Chunked standardization & scoring with optional checkpoint
# -----------------------------
def standardize_chunks(df, features, scaler, chunk_rows, checkpoint_dir=None):
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_probs = os.path.join(checkpoint_dir, "probs.npy")
        if os.path.exists(ckpt_probs):
            # We already have probabilities on disk; skip standardization
            return None, True
    n = len(df)
    cols_numeric = df[features]
    X_std_parts = []
    steps = max(1, int(np.ceil(n / chunk_rows)))
    println("Scoring probabilities…")
    sys.stdout.write("  • chunks [")
    sys.stdout.flush()
    for i in range(steps):
        lo = i * chunk_rows
        hi = min((i + 1) * chunk_rows, n)
        X_chunk = cols_numeric.iloc[lo:hi].to_numpy(float)
        Xs = scaler.transform(X_chunk)
        X_std_parts.append(Xs)
        sys.stdout.write("█")
        sys.stdout.flush()
    sys.stdout.write(f"] {steps}/{steps}\n")
    X_std = np.vstack(X_std_parts)
    return X_std, False

def predict_probs(model, X_std, checkpoint_dir=None):
    if checkpoint_dir:
        probs_path = os.path.join(checkpoint_dir, "probs.npy")
        if X_std is None:
            # resume
            if os.path.exists(probs_path):
                return np.load(probs_path)
            raise RuntimeError("No standardized features and no checkpoint probabilities found.")
        # fresh compute
        p = model.predict_proba(X_std)[:, 1]
        np.save(probs_path, p)
        return p
    # no checkpoint
    return model.predict_proba(X_std)[:, 1]

# -----------------------------
# Motion-aware label construction (fixed)
# -----------------------------
def future_max_label_by_point(df, target_col, hours):
    """
    For each (lat,lon), compute future max of target over [t, t+hours] AT THE SAME CELL.
    Avoids 'time' index/column ambiguity by constructing a Series with DatetimeIndex,
    rolling in time, then aligning back to the original row order.
    """
    # Ensure well-ordered
    df = df.sort_values(["lat","lon","time"], kind="mergesort")

    def _per_cell(g):
        # g has columns: lat, lon, time, target
        g = g.sort_values("time")
        s = pd.Series(g[target_col].astype(int).to_numpy(), index=pd.to_datetime(g["time"].to_numpy()))
        # future max via reverse + time-based rolling
        r = s.iloc[::-1].rolling(f"{hours}h", min_periods=1).max().iloc[::-1]
        # return in the same row order as g (by original index)
        out = pd.Series(r.values, index=g.index)
        return out

    # group and apply without creating extra 'time' levels
    out = df.groupby(["lat","lon"], sort=False, group_keys=False).apply(_per_cell)
    # Align to original incoming df index order
    out = out.reindex(df.index)
    return out.fillna(0).astype(int).to_numpy()

def _dilate_2d_bool(grid, radius):
    """Chebyshev (square) neighborhood dilation radius (fast & simple)."""
    if radius <= 0: return grid
    H, W = grid.shape
    pad = radius
    padded = np.pad(grid, ((pad,pad),(pad,pad)), mode="constant", constant_values=False)
    out = np.zeros_like(grid, dtype=bool)
    for di in range(-radius, radius+1):
        for dj in range(-radius, radius+1):
            si = di + pad
            sj = dj + pad
            out |= padded[si:si+H, sj:sj+W]
    return out

def apply_neighbor_dilation(df, labels_vec, radius):
    """
    Per-hour neighbor dilation in grid space.
    df must contain columns: time, lat, lon (hourly slices forming full mesh).
    labels_vec: 0/1 array aligned with df rows.
    """
    if radius <= 0:
        return labels_vec

    # Build grid index maps (stable across hours)
    lats = np.sort(df["lat"].unique())
    lons = np.sort(df["lon"].unique())
    lat_to_i = {v:i for i,v in enumerate(lats)}
    lon_to_j = {v:i for i,v in enumerate(lons)}
    i_idx = df["lat"].map(lat_to_i).to_numpy()
    j_idx = df["lon"].map(lon_to_j).to_numpy()

    out = np.zeros_like(labels_vec, dtype=np.int8)

    # Iterate by hour without making time both index and column
    for t, g in df.groupby("time", sort=False, group_keys=False):
        rows = g.index.to_numpy()
        sub_i = i_idx[rows]
        sub_j = j_idx[rows]
        sub_y = labels_vec[rows].astype(bool)

        H, W = len(lats), len(lons)
        grid = np.zeros((H, W), dtype=bool)
        grid[sub_i, sub_j] = sub_y
        grid_d = _dilate_2d_bool(grid, radius)
        out[rows] = grid_d[sub_i, sub_j].astype(np.int8)

    return out.astype(int)

# -----------------------------
# Optional add-ons (non-fatal)
# -----------------------------
def optional_metrics(df):
    hooks = []
    if "rain_mm" in df.columns:
        hooks.append(("rain_mm", df["rain_mm"].describe(percentiles=[.05,.5,.95]).to_dict()))
    if "lightning_count" in df.columns:
        hooks.append(("lightning_count", df["lightning_count"].describe(percentiles=[.05,.5,.95]).to_dict()))
    if hooks:
        println("\n[Optional feature percentiles]")
        for name, stats in hooks:
            p05 = stats.get("5%", np.nan)
            p50 = stats.get("50%", np.nan)
            p95 = stats.get("95%", np.nan)
            println(f"  {name:15s} p05={p05:.3f}  p50={p50:.3f}  p95={p95:.3f}")
    else:
        println("\n[Optional features] rainfall/lightning not found (skipping).")

# -----------------------------
# Metrics
# -----------------------------
def evaluate(y, p, tag):
    auc   = roc_auc_score(y, p)
    prauc = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    pos   = int(y.sum())
    n     = len(y)
    println(f"{tag}  →  AUC={auc:.3f}  PRAUC={prauc:.3f}  Brier={brier:.3f}  Pos={pos:,}/{n:,}")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Lead-time evaluation with motion-aware labels.")
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--model",    required=True)
    ap.add_argument("--target",   required=True, choices=["storm","near_storm","pregen"])
    ap.add_argument("--lead-hours", nargs="+", type=int, required=True)
    ap.add_argument("--neighbor-radius", type=int, default=1, help="per-hour spatial dilation (cells)")
    ap.add_argument("--advect-cells",    type=int, default=0, help="extra dilation to approximate drift")
    ap.add_argument("--chunk-rows",      type=int, default=2_000_000)
    ap.add_argument("--checkpoint-dir",  default=None)
    args = ap.parse_args()

    println("== Lead-Time Evaluation (Motion-aware) ==")
    println(f"Labelled : {args.labelled}")
    println(f"Model    : {args.model}")
    println(f"Target   : {args.target}")
    println(f"Leads    : {args.lead_hours}")
    println(f"Neighbor : radius={args.neighbor_radius}  Advect-cells={args.advect_cells}")

    # Load data
    df = pd.read_csv(args.labelled, parse_dates=["time"])
    model, scaler, features = load_model(args.model)

    need = features + [args.target, "lat", "lon", "time"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Labelled file missing columns: {missing}")

    # Drop NaNs only in required columns (keep the grid intact otherwise)
    df = df.dropna(subset=features + [args.target]).reset_index(drop=True)

    optional_metrics(df)

    n = len(df)
    base = df[args.target].astype(int).to_numpy()
    println(f"Rows evaluated: {n:,}  Positives (coincident {args.target}): {base.sum():,}")

    # Standardize + predict (with checkpoint)
    X_std, had_ckpt = standardize_chunks(df, features, scaler, args.chunk_rows, args.checkpoint_dir)
    if had_ckpt:
        println("Probabilities already scored (checkpoint).")
    p = predict_probs(model, X_std, args.checkpoint_dir)

    println("[COINCIDENT]")
    evaluate(base, p, tag="COINCIDENT")

    total_radius = max(0, int(args.neighbor_radius) + int(args.advect_cells))

    for h in args.lead_hours:
        println(f"Preparing lead +{h}h labels …")
        y_future = future_max_label_by_point(df[["lat","lon","time",args.target]].copy(), args.target, hours=h)
        y_future_dil = apply_neighbor_dilation(df[["time","lat","lon"]].copy(), y_future, radius=total_radius)
        evaluate(y_future_dil, p, tag=f"Lead +{h}h")

    # Simple overlap sanity check for last two leads
    if len(args.lead_hours) >= 2:
        h1, h2 = args.lead_hours[-2], args.lead_hours[-1]
        y1 = future_max_label_by_point(df[["lat","lon","time",args.target]].copy(), args.target, hours=h1)
        y2 = future_max_label_by_point(df[["lat","lon","time",args.target]].copy(), args.target, hours=h2)
        y1d = apply_neighbor_dilation(df[["time","lat","lon"]].copy(), y1, radius=total_radius)
        y2d = apply_neighbor_dilation(df[["time","lat","lon"]].copy(), y2, radius=total_radius)
        inter = np.logical_and(y1d>0, y2d>0).sum()
        union = np.logical_or (y1d>0, y2d>0).sum()
        jacc  = (inter/union) if union>0 else 1.0
        println(f"[Sanity] Overlap {h1}h vs {h2}h → Jaccard={jacc:.3f}  Pos{h1}={y1d.sum():,}  Pos{h2}={y2d.sum():,}")

if __name__ == "__main__":
    main()