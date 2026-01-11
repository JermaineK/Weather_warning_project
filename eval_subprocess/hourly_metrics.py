#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hourly_metrics.py
Aggregate per-hour alert coverage and simple cluster stats
across one or more alert CSVs.

Required columns in inputs:
  time, lat, lon, <flag_col>

Optional:
  <prob_col> (configurable; default 'prob')
"""

import argparse, glob, math, sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from utils import join_audit

# -------------------- helpers --------------------

def km_per_deg_lat() -> float:
    return 111.32

def km_per_deg_lon(lat_deg: float) -> float:
    return 111.32 * math.cos(math.radians(lat_deg))

def normalize_lon_series(s: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if mode == "none":
        return x
    if mode == "0..360":
        return (x % 360 + 360) % 360
    return ((x + 180) % 360) - 180  # -180..180

def parse_area(aoi: str | None):
    if not aoi:
        return None
    latN, lonW, latS, lonE = [float(x.strip()) for x in aoi.split(",")]
    return latN, lonW, latS, lonE

def infer_regular_grid(df_ll: pd.DataFrame, tol_frac: float = 0.05):
    """
    Infer a regular lat/lon grid from unique coords.
    Returns (is_regular, lats, lons, dlat, dlon)
    """
    lats = np.sort(df_ll["lat"].unique())
    lons = np.sort(df_ll["lon"].unique())
    if lats.size < 2 or lons.size < 2:
        return (False, None, None, None, None)

    dlat = np.diff(lats)
    dlon = np.diff(lons)
    mlat = np.median(dlat) if dlat.size else 0.0
    mlon = np.median(dlon) if dlon.size else 0.0
    if mlat <= 0 or mlon <= 0:
        return (False, None, None, None, None)

    ok_lat = np.all(np.abs(dlat - mlat) <= tol_frac * max(mlat, 1e-12))
    ok_lon = np.all(np.abs(dlon - mlon) <= tol_frac * max(mlon, 1e-12))
    if ok_lat and ok_lon:
        return (True, lats, lons, float(mlat), float(mlon))
    return (False, None, None, None, None)

def _snap_to_grid(vals: np.ndarray, grid: np.ndarray, half_step: float) -> np.ndarray:
    """Snap values to nearest grid point if within half_step; else NaN."""
    idx = np.searchsorted(grid, vals, side="left")
    idx = np.clip(idx, 1, len(grid)-1)
    left = grid[idx - 1]
    right = grid[idx]
    choose_right = (right - vals) < (vals - left)
    near = np.where(choose_right, right, left)
    ok = np.abs(near - vals) <= half_step + 1e-12
    out = np.where(ok, near, np.nan)
    return out

def cluster_stats_grid(active_df: pd.DataFrame, lats, lons, dlat, dlon):
    """
    4-neighbor components on a regular grid with float jitter tolerance.
    Returns (n_clusters, mean_area_km2, median_area_km2).
    """
    if active_df.empty:
        return 0, float("nan"), float("nan")

    # Snap coordinates to grid (within half a step)
    lat_snap = _snap_to_grid(active_df["lat"].to_numpy(float), lats, dlat/2)
    lon_snap = _snap_to_grid(active_df["lon"].to_numpy(float), lons, dlon/2)
    ok = np.isfinite(lat_snap) & np.isfinite(lon_snap)
    if not np.any(ok):
        return 0, float("nan"), float("nan")

    lat_to_i = {v: i for i, v in enumerate(lats)}
    lon_to_j = {v: j for j, v in enumerate(lons)}
    active = {(lat_to_i[la], lon_to_j[lo]) for la, lo in zip(lat_snap[ok], lon_snap[ok])
              if la in lat_to_i and lo in lon_to_j}
    if not active:
        return 0, float("nan"), float("nan")

    lat_mid = float(np.mean(lats))
    dy_km = km_per_deg_lat() * dlat
    dx_km = km_per_deg_lon(lat_mid) * dlon
    cell_area = max(dy_km * dx_km, 0.0)

    visited = set()
    sizes = []
    for cell in active:
        if cell in visited:
            continue
        stack = [cell]
        visited.add(cell)
        n_cells = 0
        while stack:
            i, j = stack.pop()
            n_cells += 1
            for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
                nb = (i+di, j+dj)
                if nb in active and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        sizes.append(n_cells * cell_area)

    arr = np.array(sizes, dtype=float)
    return len(arr), float(np.nanmean(arr)), float(np.nanmedian(arr))

def read_many(paths, cols_needed, prob_col):
    dfs = []
    for p in paths:
        try:
            usecols = [c for c in cols_needed if c not in ("prob", prob_col)] + [prob_col]
            usecols = list(dict.fromkeys(usecols))  # preserve order, dedupe
            if str(p).lower().endswith((".parquet", ".pq", ".pqt")):
                df = pd.read_parquet(p, columns=None)
            else:
                df = pd.read_csv(p, usecols=lambda c: (c in usecols) or (c == "prob"), low_memory=False)
            if "prob" in df.columns and prob_col not in df.columns:
                df.rename(columns={"prob": prob_col}, inplace=True)
            df["_src"] = p
            dfs.append(df)
        except Exception as e:
            print(f"[warn] failed to read {p}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def expand_alert_patterns(patterns):
    out = []
    for pat in patterns:
        out.extend(glob.glob(pat))
    return sorted(set(out))


def _write_output(df: pd.DataFrame, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suf = p.suffix.lower()
    if suf in (".parquet", ".pq", ".pqt"):
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False)

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser(description="Compute hourly alert metrics with optional cluster stats.")
    ap.add_argument("--alerts", nargs="+", required=False, default=None, help="Glob pattern(s) to alerts CSVs.")
    ap.add_argument("--flag-col", default="alert_final", help="Binary alert flag column.")
    ap.add_argument("--prob-col", default="prob_viable", help="Probability column name if present (default: prob_viable).")
    ap.add_argument("--normalize-lon", choices=["none","-180..180","0..360"], default="-180..180",
                    type=str,
                    help="Normalize longitude before clustering/metrics.")
    ap.add_argument("--area", default=None, help='Optional AOI "latN,lonW,latS,lonE" after lon normalization.')
    ap.add_argument("--tag", default=None, help="Short tag to include in output (e.g. 'thr' or 'den').")
    ap.add_argument("--out", required=False, default=None, help="Output CSV path for hourly KPIs.")
    ap.add_argument("--run-name", default=None, help="Optional run name to auto-fill tag/paths (alerts_<run>_*).")
    ap.add_argument("--skip-if-exists", action="store_true", help="Skip work if output already exists.")
    # Chunk hints (accepted for compatibility; not used)
    ap.add_argument("--chunk-rows", type=int, default=None, help="Accepted for compatibility; not used.")
    ap.add_argument("--chunksize", type=int, default=None, help="Alias for --chunk-rows.")
    ap.add_argument("--parquet-rows", type=int, default=None, help="Accepted for compatibility; not used.")
    # preprocess normalize-lon to handle tokens like "-180..180"
    argv = []
    skip = False
    raw = sys.argv[1:]
    for i, tok in enumerate(raw):
        if skip:
            skip = False
            continue
        if tok == "--normalize-lon" and i + 1 < len(raw):
            val = raw[i + 1].strip()
            argv.append(f"--normalize-lon={val}")
            skip = True
        elif tok.startswith("--normalize-lon="):
            lhs, rhs = tok.split("=", 1)
            argv.append(f"{lhs}={rhs.strip()}")
        else:
            argv.append(tok)
    args = ap.parse_args(argv)

    if args.alerts is None:
        if args.run_name:
            args.alerts = [f"results/alerts/alerts_{args.run_name}_*.csv.gz"]
        else:
            raise SystemExit("--alerts is required (or provide --run-name for defaults).")

    if args.tag is None:
        args.tag = args.run_name or "run"
    if args.out is None:
        args.out = (
            f"results/metrics/{args.run_name}_hourly_metrics.csv"
            if args.run_name else "results/metrics/hourly_metrics.csv"
        )
    if args.skip_if_exists and Path(args.out).exists():
        print(f"[skip] output already exists: {args.out}")
        return

    paths = expand_alert_patterns(args.alerts)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    cols_needed = {"time", "lat", "lon", args.flag_col}
    df = read_many(paths, cols_needed, args.prob_col) if paths else pd.DataFrame()

    if df.empty:
        # Empty skeleton
        empty = pd.DataFrame(columns=[
            "_hour","_tag","rows","active","coverage",
            "n_clusters","mean_cluster_area_km2","median_cluster_area_km2",
            "mean_prob_active","max_prob_hour",
            "n_files","hour_start"
        ])
        _write_output(empty, args.out)
        print(f"[write] {args.out} (no rows)")
        return

    missing = cols_needed - set(df.columns)
    if missing:
        raise ValueError(f"Input must contain {sorted(cols_needed)}; missing {sorted(missing)}")

    # Normalize lon and optional AOI crop before any grouping
    df["lon"] = normalize_lon_series(df["lon"], args.normalize_lon)
    if args.area:
        latN, lonW, latS, lonE = parse_area(args.area)
        df = df.loc[(df["lat"] <= latN) & (df["lat"] >= latS) &
                    (df["lon"] >= lonW) & (df["lon"] <= lonE)].reset_index(drop=True)

    # Normalize time -> UTC-naive floor-hour
    t = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["_hour"] = t.dt.tz_convert(None).dt.floor("h")

    # Coerce flags to numeric 0/1
    df[args.flag_col] = pd.to_numeric(df[args.flag_col], errors="coerce").fillna(0).astype(int)

    # Basic hourly aggregations
    base = (df.groupby("_hour")
              .agg(rows=("time","size"),
                   active=(args.flag_col,"sum"),
                   n_files=("_src","nunique"))
              .reset_index())
    base["_tag"] = args.tag
    base["coverage"] = base["active"] / base["rows"]

    # Probability summaries if present
    if args.prob_col in df.columns:
        act = df.loc[df[args.flag_col] > 0, ["_hour", args.prob_col]].copy()
        prob_mean = act.groupby("_hour")[args.prob_col].mean().rename("mean_prob_active")
        prob_max  = df.groupby("_hour")[args.prob_col].max().rename("max_prob_hour")
        left_df = base
        prob_mean_df = prob_mean.reset_index()
        base = base.merge(prob_mean_df, on="_hour", how="left")
        left_dupe = int(left_df.duplicated(subset=["_hour"]).sum())
        right_dupe = int(prob_mean_df.duplicated(subset=["_hour"]).sum())
        unmatched = join_audit.estimate_unmatched_keys(left_df, prob_mean_df, ["_hour"])
        entry = join_audit.build_entry(
            step="eval.hourly-metrics.prob-mean-merge",
            keys=["_hour"],
            join_type="left",
            left_rows=len(left_df),
            right_rows=len(prob_mean_df),
            out_rows=len(base),
            left_dupe_keys=left_dupe,
            right_dupe_keys=right_dupe,
            left_key_count=unmatched.get("left_key_count"),
            right_key_count=unmatched.get("right_key_count"),
            left_unmatched_keys=unmatched.get("left_unmatched_keys"),
            right_unmatched_keys=unmatched.get("right_unmatched_keys"),
            unmatched_sampled=unmatched.get("unmatched_sampled"),
            extra={},
        )
        join_audit.append_entry(join_audit.default_path(), entry)
        left_df = base
        prob_max_df = prob_max.reset_index()
        base = base.merge(prob_max_df,  on="_hour", how="left")
        left_dupe = int(left_df.duplicated(subset=["_hour"]).sum())
        right_dupe = int(prob_max_df.duplicated(subset=["_hour"]).sum())
        unmatched = join_audit.estimate_unmatched_keys(left_df, prob_max_df, ["_hour"])
        entry = join_audit.build_entry(
            step="eval.hourly-metrics.prob-max-merge",
            keys=["_hour"],
            join_type="left",
            left_rows=len(left_df),
            right_rows=len(prob_max_df),
            out_rows=len(base),
            left_dupe_keys=left_dupe,
            right_dupe_keys=right_dupe,
            left_key_count=unmatched.get("left_key_count"),
            right_key_count=unmatched.get("right_key_count"),
            left_unmatched_keys=unmatched.get("left_unmatched_keys"),
            right_unmatched_keys=unmatched.get("right_unmatched_keys"),
            unmatched_sampled=unmatched.get("unmatched_sampled"),
            extra={},
        )
        join_audit.append_entry(join_audit.default_path(), entry)
    else:
        base["mean_prob_active"] = np.nan
        base["max_prob_hour"]    = np.nan

    # Cluster metrics per hour (robust)
    clusters = []
    for hr, g in df.groupby("_hour", sort=True):
        reg, lats, lons, dlat, dlon = infer_regular_grid(g[["lat","lon"]].drop_duplicates())
        if not reg:
            n_c, mean_a, med_a = 0, float("nan"), float("nan")
        else:
            n_c, mean_a, med_a = cluster_stats_grid(
                g.loc[g[args.flag_col] > 0, ["lat","lon"]],
                lats, lons, dlat, dlon
            )
        clusters.append({"_hour": hr, "_tag": args.tag,
                         "n_clusters": n_c,
                         "mean_cluster_area_km2": mean_a,
                         "median_cluster_area_km2": med_a})
    clus = pd.DataFrame(clusters)

    left_df = base
    out = base.merge(clus, on=["_hour","_tag"], how="left")
    left_dupe = int(left_df.duplicated(subset=["_hour", "_tag"]).sum())
    right_dupe = int(clus.duplicated(subset=["_hour", "_tag"]).sum())
    unmatched = join_audit.estimate_unmatched_keys(left_df, clus, ["_hour", "_tag"])
    entry = join_audit.build_entry(
        step="eval.hourly-metrics.cluster-merge",
        keys=["_hour", "_tag"],
        join_type="left",
        left_rows=len(left_df),
        right_rows=len(clus),
        out_rows=len(out),
        left_dupe_keys=left_dupe,
        right_dupe_keys=right_dupe,
        left_key_count=unmatched.get("left_key_count"),
        right_key_count=unmatched.get("right_key_count"),
        left_unmatched_keys=unmatched.get("left_unmatched_keys"),
        right_unmatched_keys=unmatched.get("right_unmatched_keys"),
        unmatched_sampled=unmatched.get("unmatched_sampled"),
        extra={},
    )
    join_audit.append_entry(join_audit.default_path(), entry)
    out["hour_start"] = out["_hour"].dt.strftime("%Y-%m-%d %H:00:00")

    cols = ["_hour","_tag","rows","active","coverage",
            "n_clusters","mean_cluster_area_km2","median_cluster_area_km2",
            "mean_prob_active","max_prob_hour",
            "n_files","hour_start"]
    out = out[cols].sort_values("_hour").reset_index(drop=True)

    _write_output(out, args.out)
    print(f"[write] {args.out} (rows={len(out)})")

if __name__ == "__main__":
    main()
