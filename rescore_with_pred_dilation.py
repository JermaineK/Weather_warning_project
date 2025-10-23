import argparse, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def _dilate_2d_bool(grid, radius):
    if radius <= 0: return grid
    H,W = grid.shape
    pad = radius
    padded = np.pad(grid, ((pad,pad),(pad,pad)), constant_values=False)
    out = np.zeros_like(grid, dtype=bool)
    for di in range(-radius, radius+1):
        for dj in range(-radius, radius+1):
            out |= padded[di+pad:di+pad+H, dj+pad:dj+pad+W]
    return out

def _dilate_2d_max(grid, radius):
    if radius <= 0: return grid
    H,W = grid.shape
    pad = radius
    padded = np.pad(grid, ((pad,pad),(pad,pad)), mode="edge")
    out = np.zeros_like(grid)
    for di in range(-radius, radius+1):
        for dj in range(-radius, radius+1):
            out = np.maximum(out, padded[di+pad:di+pad+H, dj+pad:dj+pad+W])
    return out

def future_max_label_by_point(df, target, hours):
    df = df.sort_values(["lat","lon","time"], kind="mergesort")
    def _per_cell(g):
        s = pd.Series(g[target].astype(int).to_numpy(), index=pd.to_datetime(g["time"].to_numpy()))
        r = s.iloc[::-1].rolling(f"{hours}h", min_periods=1).max().iloc[::-1]
        return pd.Series(r.values, index=g.index)
    out = df.groupby(["lat","lon"], sort=False, group_keys=False).apply(_per_cell)
    return out.reindex(df.index).fillna(0).astype(int).to_numpy()

def apply_neighbor_dilation_label(df, y, radius):
    if radius <= 0: return y
    lats = np.sort(df["lat"].unique()); lons = np.sort(df["lon"].unique())
    li = df["lat"].map({v:i for i,v in enumerate(lats)}).to_numpy()
    lj = df["lon"].map({v:i for i,v in enumerate(lons)}).to_numpy()
    out = np.zeros_like(y, dtype=np.int8)
    for _,g in df.groupby("time", sort=False, group_keys=False):
        idx = g.index.to_numpy()
        H,W = len(lats), len(lons)
        grid = np.zeros((H,W), dtype=bool)
        grid[li[idx], lj[idx]] = (y[idx] > 0)
        grid = _dilate_2d_bool(grid, radius)
        out[idx] = grid[li[idx], lj[idx]].astype(np.int8)
    return out.astype(int)

def apply_neighbor_dilation_pred(df, p, radius):
    if radius <= 0: return p
    lats = np.sort(df["lat"].unique()); lons = np.sort(df["lon"].unique())
    li = df["lat"].map({v:i for i,v in enumerate(lats)}).to_numpy()
    lj = df["lon"].map({v:i for i,v in enumerate(lons)}).to_numpy()
    out = np.zeros_like(p, dtype=float)
    for _,g in df.groupby("time", sort=False, group_keys=False):
        idx = g.index.to_numpy()
        H,W = len(lats), len(lons)
        grid = np.zeros((H,W), dtype=float)
        grid[li[idx], lj[idx]] = p[idx]
        grid = _dilate_2d_max(grid, radius)
        out[idx] = grid[li[idx], lj[idx]]
    return out

def evaluate(y, p, tag):
    auc   = roc_auc_score(y, p)
    prauc = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    print(f"{tag}  →  AUC={auc:.3f}  PRAUC={prauc:.3f}  Brier={brier:.3f}  Pos={int(y.sum()):,}/{len(y):,}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--probs-npy", required=True)
    ap.add_argument("--target", required=True, choices=["storm","near_storm","pregen"])
    ap.add_argument("--lead-hours", nargs="+", type=int, required=True)
    ap.add_argument("--neighbor-radius", type=int, default=1)
    ap.add_argument("--advect-cells", type=int, default=1)
    args = ap.parse_args()

    df = pd.read_csv(args.labelled, parse_dates=["time"])
    need = ["lat","lon","time", args.target]
    missing = [c for c in need if c not in df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")

    p = np.load(args.probs_npy)
    if len(p) != len(df): raise ValueError("probs.npy length does not match dataframe rows.")

    total_r = max(0, args.neighbor_radius + args.advect_cells)

    # Dilate predictions per hour to match the label geometry
    p_dil = apply_neighbor_dilation_pred(df[["time","lat","lon"]], p, radius=total_r)

    # Coincident (no temporal look-ahead), but spatially fair:
    y0 = df[args.target].astype(int).to_numpy()
    y0_dil = apply_neighbor_dilation_label(df[["time","lat","lon"]], y0, radius=total_r)
    print("[FAIR | COINCIDENT]", flush=True)
    evaluate(y0_dil, p_dil, "COINCIDENT")

    for h in args.lead_hours:
        print(f"Preparing lead +{h}h labels …", flush=True)
        y_f = future_max_label_by_point(df[["lat","lon","time",args.target]].copy(), args.target, hours=h)
        y_fd = apply_neighbor_dilation_label(df[["time","lat","lon"]], y_f, radius=total_r)
        evaluate(y_fd, p_dil, f"Lead +{h}h")

if __name__ == "__main__":
    main()