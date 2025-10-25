#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seed_starts_vs_tracks.py
Identify seed "starts" (first hour of persistent runs), cluster them into
spatial patches per hour, and match to best-track to compute lead-time metrics.

Inputs
------
--seeds : CSV or Parquet of seed cells with at least: time, lat, lon, and a 0/1 flag column.
          Defaults: time column 'time', flag column '__flag__'. If missing, tries 'alert', 'alert_final', 'flag'.
--tracks: CSV/Parquet from prepare_besttrack_intensity.py (obs_time, lat, lon, vmax, pmin[, name|sid])

Outputs (in --out-dir)
----------------------
- seed_starts_points.csv      : each grid-point start (one row per start point)
- seed_patches.csv            : hourly patches (clustered starts) with centroid, bbox, size
- seed_track_matches.csv      : seed-patch → track match with distances and lead times
- seed_summary.txt            : human-readable summary

Notes
-----
- Connectivity: 4 or 8 (default 4).
- Run detection uses consecutive hourly steps per (lat,lon). A "start" is the
  first hour of any run whose length >= --min-run-hours.
- Matching: find best-track point within radius (--radius-deg) and within
  +/- --time-tol-hours around the seed *start hour*. If matched, compute lead
  times from seed start to the same storm’s future times reaching >=25/34/50/64 kt.
"""

from __future__ import annotations
import argparse, os, math
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd


# ------------------------ IO helpers ------------------------

CANDIDATE_TIME_COLS = ["time", "time_h", "datetime", "valid_time", "forecast_time"]
CANDIDATE_FLAG_COLS = ["__flag__", "alert_final", "alert", "flag"]

def read_any(path: str, **kw) -> pd.DataFrame:
    p = str(path)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    if p.lower().endswith(".parquet"):
        return pd.read_parquet(p, **kw)
    return pd.read_csv(p, compression="infer", low_memory=False, **kw)

def to_utc_naive(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None)

def norm_lon(series: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if mode == "none":
        return x
    if mode == "0..360":
        return (x % 360 + 360) % 360
    return ((x + 180) % 360) - 180  # -180..180

def parse_area(aoi: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not aoi: return None
    parts = [p.strip() for p in str(aoi).split(",")]
    if len(parts) != 4:
        raise ValueError("--area must be 'latN,lonW,latS,lonE'")
    n, w, s, e = map(float, parts)
    return (n, w, s, e)


# ------------------------ geo/time utils ------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2.0)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2.0)**2
    return 2.0 * R * np.arcsin(np.sqrt(a))

def infer_grid_step(vals: np.ndarray) -> float:
    """Median step of unique sorted values."""
    u = np.unique(np.asarray(vals))
    if u.size < 2: return 0.0
    dif = np.diff(np.sort(u))
    dif = dif[np.isfinite(dif) & (dif > 0)]
    if dif.size == 0: return 0.0
    return float(np.median(dif))

def consecutive_runs_1d(b: np.ndarray) -> List[Tuple[int,int]]:
    """
    Given 1D boolean array b indexed hourly, return list of (start_idx, length)
    for all True-runs (consecutive Trues).
    """
    if b.size == 0:
        return []
    # Pad edges to catch transitions
    pad = np.r_[False, b, False]
    edges = np.diff(pad.astype(int))
    starts = np.where(edges == 1)[0]
    ends   = np.where(edges == -1)[0]
    return [(int(s), int(e - s)) for s, e in zip(starts, ends)]


# ------------------------ clustering ------------------------

def cluster_starts_one_hour(df_h: pd.DataFrame,
                            lat_step: float,
                            lon_step: float,
                            connectivity: int = 4) -> List[Dict]:
    """
    Input: rows with columns [time_h, lat, lon] for a single hour.
    Output: list of clusters with fields:
        { 'time_h', 'patch_id_local', 'n_cells', 'lat_cen', 'lon_cen',
          'lat_min','lat_max','lon_min','lon_max' }
    Clustering is grid-based: neighbors if Δlat<=lat_step+tol and Δlon<=lon_step+tol.
    """
    if df_h.empty:
        return []

    lat_vals = df_h["lat"].to_numpy()
    lon_vals = df_h["lon"].to_numpy()
    n = len(df_h)

    # Index by position; adjacency by geometric proximity ~ grid step (loose tolerance)
    tol_lat = lat_step * 1.05 if lat_step > 0 else 0.001
    tol_lon = lon_step * 1.05 if lon_step > 0 else 0.001

    # Build adjacency lists (O(n^2) is fine for dozens/hundreds; if thousands, could grid-hash)
    # To be safer, use a simple bin hash by rounding to nearest step.
    lat_k = np.round(lat_vals / max(lat_step, tol_lat), 3) if lat_step > 0 else np.round(lat_vals, 3)
    lon_k = np.round(lon_vals / max(lon_step, tol_lon), 3) if lon_step > 0 else np.round(lon_vals, 3)
    # map from (key_lat,key_lon) to indices
    from collections import defaultdict
    bins = defaultdict(list)
    for i, (lk, ok) in enumerate(zip(lat_k, lon_k)):
        bins[(lk, ok)].append(i)

    # Neighbor key offsets
    offs = [(-1,0),(1,0),(0,-1),(0,1)]
    if connectivity == 8:
        offs += [(-1,-1),(-1,1),(1,-1),(1,1)]

    visited = np.zeros(n, dtype=bool)
    clusters = []
    # Build quick mapping from key -> neighbors by probing surrounding keys
    def neighbors(i):
        lk, ok = lat_k[i], lon_k[i]
        ns = []
        for dy, dx in offs:
            key = (lk + dy, ok + dx)
            for j in bins.get(key, []):
                # Extra numeric guard for weird grids: enforce actual distances too
                if (abs(lat_vals[j]-lat_vals[i]) <= tol_lat + 1e-9 and
                    abs(lon_vals[j]-lon_vals[i]) <= tol_lon + 1e-9):
                    ns.append(j)
        return ns

    for i in range(n):
        if visited[i]: continue
        # BFS
        q = [i]
        visited[i] = True
        group = [i]
        while q:
            u = q.pop()
            for v in neighbors(u):
                if not visited[v]:
                    visited[v] = True
                    q.append(v)
                    group.append(v)
        lat_g = lat_vals[group]; lon_g = lon_vals[group]
        clusters.append(dict(
            time_h=df_h["time_h"].iloc[0],
            patch_id_local=len(clusters),
            n_cells=len(group),
            lat_cen=float(np.mean(lat_g)),
            lon_cen=float(np.mean(lon_g)),
            lat_min=float(np.min(lat_g)),
            lat_max=float(np.max(lat_g)),
            lon_min=float(np.min(lon_g)),
            lon_max=float(np.max(lon_g)),
        ))
    return clusters


# ------------------------ load + prepare ------------------------

def load_seeds(path: str,
               time_col: Optional[str],
               flag_col: Optional[str],
               normalize_lon_mode: str,
               area: Optional[str]) -> Tuple[pd.DataFrame, str, str]:
    df = read_any(path).replace([np.inf,-np.inf], np.nan)

    # Pick time column
    tcol = time_col or next((c for c in CANDIDATE_TIME_COLS if c in df.columns), None)
    if tcol is None:
        raise ValueError(f"{path}: need a time column (tried {CANDIDATE_TIME_COLS}).")
    t = to_utc_naive(df[tcol])
    df = df.loc[t.notna()].copy(); df[tcol] = t[t.notna()]

    # Pick flag column
    fcol = flag_col or next((c for c in CANDIDATE_FLAG_COLS if c in df.columns), None)
    if fcol is None:
        # If none present, assume "1" for all rows (treat every row as a seed hit)
        df["__flag__auto__"] = 1
        fcol = "__flag__auto__"
    df[fcol] = pd.to_numeric(df[fcol], errors="coerce").fillna(0).astype(int)

    # Numerics + lon norm
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = norm_lon(df["lon"], normalize_lon_mode)
    df = df.dropna(subset=[tcol, "lat", "lon"]).reset_index(drop=True)

    # AOI
    a = parse_area(area)
    if a:
        latN, lonW, latS, lonE = a
        df = df.loc[(df["lat"] <= latN) & (df["lat"] >= latS) &
                    (df["lon"] >= lonW) & (df["lon"] <= lonE)].reset_index(drop=True)

    # Force hourly
    df["time_h"] = df[tcol].dt.floor("h")
    return df, tcol, fcol

def load_tracks(path: str,
                time_offset_h: float = 0.0,
                normalize_lon_mode: str = "none",
                area: Optional[str] = None) -> pd.DataFrame:
    tr = read_any(path)
    time_col = "obs_time" if "obs_time" in tr.columns else ("time" if "time" in tr.columns else None)
    if time_col is None:
        raise ValueError(f"{path}: need an 'obs_time' (or 'time') column")
    for req in ["lat","lon","vmax","pmin"]:
        if req not in tr.columns:
            raise ValueError(f"{path}: missing column '{req}'")

    t = to_utc_naive(tr[time_col])
    if time_offset_h:
        t = t + pd.to_timedelta(time_offset_h, unit="h")

    out = pd.DataFrame({
        "obs_time": t,
        "lat": pd.to_numeric(tr["lat"], errors="coerce"),
        "lon": norm_lon(pd.to_numeric(tr["lon"], errors="coerce"), normalize_lon_mode),
        "vmax": pd.to_numeric(tr["vmax"], errors="coerce"),
        "pmin": pd.to_numeric(tr["pmin"], errors="coerce"),
    })
    # Attach name/sid if present (helps grouping)
    if "name" in tr.columns: out["name"] = tr["name"]
    if "sid"  in tr.columns: out["sid"]  = tr["sid"]

    out = out.dropna(subset=["obs_time","lat","lon","vmax","pmin"]).reset_index(drop=True)

    a = parse_area(area)
    if a:
        latN, lonW, latS, lonE = a
        out = out.loc[(out["lat"] <= latN) & (out["lat"] >= latS) &
                      (out["lon"] >= lonW) & (out["lon"] <= lonE)].reset_index(drop=True)

    out.sort_values("obs_time", inplace=True, ignore_index=True)
    return out


# ------------------------ seed starts ------------------------

def find_seed_starts(df: pd.DataFrame,
                     time_col: str,
                     flag_col: str,
                     min_run_hours: int = 72) -> pd.DataFrame:
    """
    For each (lat,lon), detect consecutive hourly True-runs in flag_col and
    record the FIRST hour of runs whose length >= min_run_hours as a "start".
    Returns rows: [time_h, lat, lon].
    """
    # Group by point, already hourly
    starts = []
    for (lat, lon), g in df.groupby(["lat","lon"], sort=False):
        t = g["time_h"].sort_values(kind="mergesort").unique()
        # Build a continuous hourly index for run detection
        # Map t -> dense index
        if t.size == 0:
            continue
        t_min, t_max = t.min(), t.max()
        full = pd.date_range(t_min, t_max, freq="H")
        # Reindex to full hours, fill flag with 0 where missing
        gg = g.set_index("time_h").sort_index()
        s = gg[flag_col].reindex(full).fillna(0).astype(int).to_numpy()
        runs = consecutive_runs_1d(s == 1)
        for s_idx, length in runs:
            if length >= int(min_run_hours):
                t0 = full[s_idx]  # first hour
                starts.append({"time_h": t0, "lat": float(lat), "lon": float(lon), "run_len_h": int(length)})
    return pd.DataFrame(starts).sort_values(["time_h","lat","lon"], ignore_index=True)


# ------------------------ match to tracks ------------------------

def pick_track_id(df: pd.DataFrame) -> str:
    if "sid" in df.columns: return "sid"
    if "name" in df.columns: return "name"
    return ""  # fallback: no explicit id

def match_patches_to_tracks(patches: pd.DataFrame,
                            tracks: pd.DataFrame,
                            radius_deg: float = 1.0,
                            time_tol_h: float = 6.0) -> pd.DataFrame:
    """
    For each seed patch (centroid, time_h), find any track point within radius and ±time_tol.
    If multiple, choose the one with minimal great-circle distance (km).
    Then compute lead times from seed start to the same storm reaching >= {25, 34, 50, 64} kt.
    """
    if patches.empty or tracks.empty:
        return pd.DataFrame(columns=[
            "time_h","patch_id","lat_cen","lon_cen","n_cells",
            "match_time","storm_id","storm_name","d_km_min","dt_hours",
            "lead_to_25h","lead_to_34h","lead_to_50h","lead_to_64h","max_vmax_plus7d"
        ])

    id_col = pick_track_id(tracks)
    by_hour = {t: g for t, g in tracks.groupby(tracks["obs_time"].dt.floor("h"))}

    out = []
    rad2 = radius_deg**2
    td = pd.Timedelta(hours=time_tol_h)

    for _, r in patches.iterrows():
        t0 = r["time_h"]
        lat0 = r["lat_cen"]; lon0 = r["lon_cen"]

        # search in hours [t0 - time_tol_h, t0 + time_tol_h]
        hwin = pd.date_range(t0 - td, t0 + td, freq="H")
        cands = []
        for th in hwin:
            g = by_hour.get(th)
            if g is None or g.empty:
                continue
            box = g.loc[(g["lat"] >= lat0 - radius_deg) & (g["lat"] <= lat0 + radius_deg) &
                        (g["lon"] >= lon0 - radius_deg) & (g["lon"] <= lon0 + radius_deg)]
            if box.empty:
                continue
            d2 = (box["lat"] - lat0)**2 + (box["lon"] - lon0)**2
            sel = box.loc[d2 <= rad2].copy()
            if sel.empty:
                continue
            sel["d_km"] = haversine_km(lat0, lon0, sel["lat"], sel["lon"])
            sel["dt_hours"] = (sel["obs_time"] - t0) / np.timedelta64(1, "h")
            cands.append(sel)

        if not cands:
            continue
        cand = pd.concat(cands, ignore_index=True)
        j = int(cand["d_km"].idxmin())
        best = cand.loc[j]

        # storm identifier/name
        storm_id = best.get(id_col, "")
        storm_name = best.get("name", "") if "name" in best.index else ""

        # lead-times: search same storm after t0
        if id_col:
            tstorm = tracks.loc[tracks[id_col] == storm_id].sort_values("obs_time")
        elif "name" in tracks.columns and isinstance(storm_name, str) and storm_name != "":
            tstorm = tracks.loc[tracks["name"] == storm_name].sort_values("obs_time")
        else:
            tstorm = tracks.iloc[[]].copy()

        lead_to = {25: np.nan, 34: np.nan, 50: np.nan, 64: np.nan}
        max_vmax_7d = np.nan
        if not tstorm.empty:
            post = tstorm.loc[tstorm["obs_time"] >= t0].copy()
            for thr in [25, 34, 50, 64]:
                hit = post.loc[post["vmax"] >= thr]
                if not hit.empty:
                    dt_h = (hit["obs_time"].iloc[0] - t0) / np.timedelta64(1, "h")
                    lead_to[thr] = float(dt_h)
            # window max up to +7 days from t0
            w_end = t0 + pd.Timedelta(days=7)
            inwin = tstorm.loc[(tstorm["obs_time"] >= t0) & (tstorm["obs_time"] <= w_end)]
            if not inwin.empty:
                max_vmax_7d = float(inwin["vmax"].max())

        out.append(dict(
            time_h=t0,
            patch_id=int(r["patch_id"]),
            lat_cen=float(lat0),
            lon_cen=float(lon0),
            n_cells=int(r["n_cells"]),
            match_time=best["obs_time"],
            storm_id=str(storm_id) if storm_id != "" else "",
            storm_name=str(storm_name) if storm_name != "" else "",
            d_km_min=float(best["d_km"]),
            dt_hours=float(best["dt_hours"]),
            lead_to_25h=float(lead_to[25]) if not math.isnan(lead_to[25]) else np.nan,
            lead_to_34h=float(lead_to[34]) if not math.isnan(lead_to[34]) else np.nan,
            lead_to_50h=float(lead_to[50]) if not math.isnan(lead_to[50]) else np.nan,
            lead_to_64h=float(lead_to[64]) if not math.isnan(lead_to[64]) else np.nan,
            max_vmax_plus7d=float(max_vmax_7d) if not math.isnan(max_vmax_7d) else np.nan,
        ))

# --- at the very end of match_patches_to_tracks() ---
    cols = [
        "time_h","patch_id","lat_cen","lon_cen","n_cells",
        "match_time","storm_id","storm_name","d_km_min","dt_hours",
        "lead_to_25h","lead_to_34h","lead_to_50h","lead_to_64h","max_vmax_plus7d"
    ]
    df_out = pd.DataFrame(out, columns=cols)
    if df_out.empty:
        return df_out  # nothing to sort; upstream will see "no matches"
    return df_out.sort_values(["time_h","patch_id"], ignore_index=True)

# ------------------------ main ------------------------

def main():
    ap = argparse.ArgumentParser(description="Find seed starts, cluster into patches, match to best-track, compute lead-times.")
    ap.add_argument("--seeds", required=True, help="Seed cells CSV/Parquet (time,lat,lon,flag).")
    ap.add_argument("--tracks", required=True, help="Best-track CSV/Parquet from prepare_besttrack_intensity.py.")
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--time-col", default=None, help="Seed time column (default: auto).")
    ap.add_argument("--flag-col", default=None, help="Seed flag column (default: auto; tries __flag__, alert_final, alert, flag).")
    ap.add_argument("--min-run-hours", type=int, default=72, help="Minimum run length to qualify as a seed start (default 72).")
    ap.add_argument("--connectivity", type=int, choices=[4,8], default=4, help="Connectivity for clustering (default 4).")
    ap.add_argument("--radius-deg", type=float, default=1.0, help="Radius for matching to tracks (degrees).")
    ap.add_argument("--time-tol-hours", type=float, default=6.0, help="+/- hours around seed start for matching.")
    ap.add_argument("--normalize-lon", choices=["none","-180..180","0..360"], default="none", help="Normalize longitudes for BOTH inputs.")
    ap.add_argument("--area", default=None, help="Optional AOI 'latN,lonW,latS,lonE' after lon normalization.")
    ap.add_argument("--track-time-offset-hours", type=float, default=0.0, help="Shift all track times by this many hours.")
    ap.add_argument("--save-parquet", action="store_true", help="Also write Parquet copies of outputs.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load
    seeds, tcol, fcol = load_seeds(args.seeds, args.time_col, args.flag_col,
                                   args.normalize_lon, args.area)
    tracks = load_tracks(args.tracks,
                         time_offset_h=args.track_time_offset_hours,
                         normalize_lon_mode=args.normalize_lon,
                         area=args.area)

    # Quick domain stats
    tmin, tmax = seeds["time_h"].min(), seeds["time_h"].max()
    print(f"[seeds] rows={len(seeds):,}  hours={seeds['time_h'].nunique():,}  "
          f"points={seeds.drop_duplicates(['lat','lon']).shape[0]:,}  "
          f"time={tmin}→{tmax}  lat={seeds['lat'].min():.3f}..{seeds['lat'].max():.3f}  "
          f"lon={seeds['lon'].min():.3f}..{seeds['lon'].max():.3f}  flag={fcol}")

    # 2) Seed starts (per point)
    starts = find_seed_starts(seeds, tcol, fcol, min_run_hours=int(args.min_run_hours))
    if starts.empty:
        print("[seed] no starts found with current min-run-hours.")
    else:
        starts["issue_hour"] = starts["time_h"]  # alias for readability
    # Save
    p_points = out_dir / "seed_starts_points.csv"
    starts.to_csv(p_points, index=False)
    if args.save_parquet:
        starts.to_parquet(out_dir / "seed_starts_points.parquet", index=False)

    # 3) Cluster starts into patches per hour
    patches_rows = []
    lat_step = infer_grid_step(seeds["lat"].values)
    lon_step = infer_grid_step(seeds["lon"].values)
    for t, g in starts.groupby("time_h", sort=True):
        clusters = cluster_starts_one_hour(g[["time_h","lat","lon"]], lat_step, lon_step,
                                           connectivity=int(args.connectivity))
        for c in clusters:
            patches_rows.append(dict(
                time_h=c["time_h"],
                patch_id=len(patches_rows),
                n_cells=c["n_cells"],
                lat_cen=c["lat_cen"], lon_cen=c["lon_cen"],
                lat_min=c["lat_min"], lat_max=c["lat_max"],
                lon_min=c["lon_min"], lon_max=c["lon_max"],
            ))
    patches = pd.DataFrame(patches_rows).sort_values(["time_h","patch_id"], ignore_index=True)
    p_patches = out_dir / "seed_patches.csv"
    patches.to_csv(p_patches, index=False)
    if args.save_parquet:
        patches.to_parquet(out_dir / "seed_patches.parquet", index=False)

    # 4) Match patches → tracks and compute lead times
    matches = match_patches_to_tracks(patches, tracks,
                                      radius_deg=float(args.radius_deg),
                                      time_tol_h=float(args.time_tol_hours))
    p_matches = out_dir / "seed_track_matches.csv"
    matches.to_csv(p_matches, index=False)
    if args.save_parquet:
        matches.to_parquet(out_dir / "seed_track_matches.parquet", index=False)

    # 5) Summary
    def _q(a, q): return float(np.nanquantile(np.asarray(a, dtype=float), q)) if len(a) else float("nan")

    n_points = seeds.drop_duplicates(["lat","lon"]).shape[0]
    n_hours  = seeds["time_h"].nunique()
    n_starts = len(starts)
    n_patches = len(patches)
    n_matched = len(matches)

    # basic lead stats
    def _lead_stats(col):
        s = matches[col].dropna()
        if s.empty: return "n=0"
        return f"n={len(s)}  median={s.median():.1f}h  p25={_q(s,0.25):.1f}h  p75={_q(s,0.75):.1f}h  max={s.max():.1f}h"

    txt = out_dir / "seed_summary.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write("Seed–Track Analysis Summary\n")
        f.write("------------------------------------------------------------\n")
        f.write(f"Seeds file  : {args.seeds}\n")
        f.write(f"Tracks file : {args.tracks}\n")
        f.write(f"Out dir     : {out_dir}\n\n")
        f.write(f"Rows(seeds) : {len(seeds):,}\n")
        f.write(f"Unique pts  : {n_points:,}\n")
        f.write(f"Hours       : {n_hours:,}\n")
        f.write(f"Time span   : {seeds['time_h'].min()} → {seeds['time_h'].max()}\n")
        f.write(f"Lat range   : {seeds['lat'].min():.3f} .. {seeds['lat'].max():.3f}\n")
        f.write(f"Lon range   : {seeds['lon'].min():.3f} .. {seeds['lon'].max():.3f}  (normalize_lon={args.normalize_lon})\n")
        f.write(f"AOI         : {args.area or '(none)'}\n")
        f.write(f"Min run (h) : {args.min_run_hours}\n")
        f.write(f"Connectivity: {args.connectivity}\n")
        f.write(f"Match rad   : {args.radius_deg}°  time tol ±{args.time_tol_hours}h\n\n")
        f.write(f"Seed starts (points): {n_starts:,}\n")
        f.write(f"Seed patches        : {n_patches:,}\n")
        f.write(f"Matched patches     : {n_matched:,}\n\n")
        if n_matched:
            f.write("Lead-time stats (from seed start to threshold):\n")
            f.write(f"  ≥25 kt : { _lead_stats('lead_to_25h') }\n")
            f.write(f"  ≥34 kt : { _lead_stats('lead_to_34h') }\n")
            f.write(f"  ≥50 kt : { _lead_stats('lead_to_50h') }\n")
            f.write(f"  ≥64 kt : { _lead_stats('lead_to_64h') }\n\n")
            f.write("Distance/time to matched point:\n")
            f.write(f"  d_km_min (median): {matches['d_km_min'].median():.1f} km   (p90={_q(matches['d_km_min'],0.9):.1f})\n")
            f.write(f"  |dt_hours| (median): {matches['dt_hours'].abs().median():.1f} h  (p90={_q(matches['dt_hours'].abs(),0.9):.1f})\n")
        else:
            f.write("No matches found; consider larger --radius-deg and/or --time-tol-hours, or check lon normalization.\n")

    print(f"[done] Wrote:\n  - {p_points}\n  - {p_patches}\n  - {p_matches}\n  - {txt}")

if __name__ == "__main__":
    main()