#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seed_proto_tracks_outcomes.py
Link hourly seeds into proto-tracks, compute simple structure features,
(optionally) join CAPE/CIN/T2M, and label each track by future storm outcome
against IBTrACS, with faster KD-tree matching and correct conditional rates.

Outputs:
  {out-dir}/{run-name}_track_points.csv[.parquet]
  {out-dir}/{run-name}_tracks.csv[.parquet]
  {out-dir}/{run-name}_conversion_rates.csv[.parquet]

Key options:
  --aoi               Optional AOI crop "latN,lonW,latS,lonE" (applied post norm-lon)
  --write-parquet     Also write Parquet copies (CSV always written)
  --time-format       Optional strptime/strftime format for non-standard time columns
"""

from __future__ import annotations

import argparse, os, math, sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from utils import join_audit

# Optional speedup for storm matching + neighbor counts
try:
    from sklearn.neighbors import KDTree, BallTree
    _HAVE_SK = True
    _HAVE_BALLTREE = True
except Exception:
    try:
        from sklearn.neighbors import BallTree  # type: ignore
        KDTree = None  # type: ignore
        _HAVE_SK = False
        _HAVE_BALLTREE = True
    except Exception:
        KDTree = None  # type: ignore
        BallTree = None  # type: ignore
        _HAVE_SK = False
        _HAVE_BALLTREE = False

# ------------------ I/O + common helpers ------------------

def read_any(p: str, **kw) -> pd.DataFrame:
    p = str(p)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    if p.lower().endswith((".parquet",".pq",".pqt")):
        return pd.read_parquet(p, **kw)
    return pd.read_csv(p, low_memory=False, **kw)

def read_csv_chunked(path: str, chunk_rows: int, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Read CSV in chunks; if max_rows is set, stop once that many rows have been gathered.
    """
    if chunk_rows and chunk_rows > 0 and (path.lower().endswith(".csv") or path.lower().endswith(".csv.gz")):
        parts = []
        total = 0
        for chunk in pd.read_csv(path, low_memory=False, compression="infer", chunksize=int(chunk_rows)):
            parts.append(chunk)
            total += len(chunk)
            if max_rows and total >= max_rows:
                break
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
        if max_rows and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=42)
        return df
    return read_any(path)

def to_utc_naive(series: pd.Series, fmt: Optional[str] = None) -> pd.Series:
    raw = series.astype(str).str.strip().str.replace("Z","",regex=False)
    if fmt:
        t = pd.to_datetime(raw, format=fmt, utc=True, errors="coerce")
    else:
        t = pd.to_datetime(raw, utc=True, errors="coerce")
    return t.dt.tz_convert(None)

def norm_lon(x: pd.Series, mode: str) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce")
    if mode == "none":
        return v
    if mode == "0..360":
        return (v % 360 + 360) % 360
    return ((v + 180) % 360) - 180  # default -180..180

def parse_area(aoi: Optional[str]) -> Optional[Tuple[float,float,float,float]]:
    if not aoi:
        return None
    latN, lonW, latS, lonE = [float(s.strip()) for s in aoi.split(",")]
    return latN, lonW, latS, lonE

def crop_area(df: pd.DataFrame, aoi: Tuple[float,float,float,float]) -> pd.DataFrame:
    latN, lonW, latS, lonE = aoi
    return df.loc[(df["lat"] <= latN) & (df["lat"] >= latS) &
                  (df["lon"] >= lonW) & (df["lon"] <= lonE)].copy()

def hav_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1r = np.radians(lat1); lat2r = np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2.0)**2
    return 2.0 * R * np.arcsin(np.sqrt(a))

def pick_time_col(df: pd.DataFrame) -> str:
    for c in ["time","time_h","valid_time","datetime"]:
        if c in df.columns:
            return c
    raise ValueError("Seeds need a time column (time/time_h/valid_time/datetime).")

def pick_prob_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["prob","prob_max","score","max_prob_hour","p"]:
        if c in df.columns:
            return c
    return None

def pick_cape_cols(df: pd.DataFrame) -> Dict[str,str]:
    caps: Dict[str,str] = {}
    for want, cands in {
        "cape": ["cape","CAPE","cape_m2s2"],
        "cin" : ["cin","CIN","cin_m2s2"],
        "t2m" : ["t2m","T2M","t","t2m_K","t2m_C"],
    }.items():
        for c in cands:
            if c in df.columns:
                caps[want] = c; break
    return caps

# ------------------ structure features ------------------

def structure_per_hour(df_hour: pd.DataFrame, radius_km: float = 100.0) -> pd.DataFrame:
    """
    Count neighbours within radius_km for one hour. Uses haversine KDTree when
    sklearn is available; falls back to O(n^2) if not. If radius_km<=0, return
    zeros to avoid heavy computation.
    """
    if df_hour.empty:
        return df_hour.copy()
    if radius_km <= 0:
        out = df_hour.copy()
        out["nbr_100km"] = 0.0
        return out

    latv = df_hour["lat"].to_numpy()
    lonv = df_hour["lon"].to_numpy()
    n = len(df_hour)

    if _HAVE_BALLTREE:
        # BallTree supports haversine directly
        coords = np.c_[np.radians(latv), np.radians(lonv)]
        tree = BallTree(coords, metric="haversine")
        rad = radius_km / 6371.0  # Earth radius ~6371 km
        nn = tree.query_radius(coords, r=rad, count_only=True) - 1  # exclude self
    else:
        nn = np.zeros(n, dtype=float)
        for i in range(n):
            d = hav_km(latv[i], lonv[i], latv, lonv)
            nn[i] = np.sum((d <= radius_km) & (d > 0))
    out = df_hour.copy()
    out["nbr_100km"] = nn
    return out


def _structure_counts(latv: np.ndarray, lonv: np.ndarray, radius_km: float) -> np.ndarray:
    """
    Return neighbor counts for a single hour without building intermediate DataFrames.
    This mirrors structure_per_hour but is memory-lean for large hour counts.
    """
    n = len(latv)
    if n == 0:
        return np.zeros(0, dtype=float)
    if radius_km <= 0:
        return np.zeros(n, dtype=float)

    if _HAVE_BALLTREE:
        coords = np.c_[np.radians(latv), np.radians(lonv)]
        tree = BallTree(coords, metric="haversine")
        rad = radius_km / 6371.0
        nn = tree.query_radius(coords, r=rad, count_only=True) - 1
        return nn.astype(float)

    nn = np.zeros(n, dtype=float)
    for i in range(n):
        d = hav_km(latv[i], lonv[i], latv, lonv)
        nn[i] = np.sum((d <= radius_km) & (d > 0))
    return nn

# ------------------ proto-track linker ------------------

def link_tracks_slow(
    seeds: pd.DataFrame,
    link_radius_km: float = 75.0,
    max_gap_hours: int = 1,
) -> pd.DataFrame:
    """
    Greedy forward-only linker with (gap, distance) tie-breaker.
    Adds: track_id, step_idx, gap_count.
    """
    df = seeds.sort_values("time").reset_index(drop=True).copy()
    df["track_id"] = -1
    df["step_idx"] = -1
    df["gap_count"] = 0

    by_hour = {h: g.reset_index() for h, g in df.groupby("time")}
    hours = sorted(by_hour.keys())

    next_tid = 1
    for h in hours:
        G = by_hour[h]
        for _, row in G.iterrows():
            idx = int(row["index"])
            if df.at[idx, "track_id"] != -1:
                continue
            # start a track
            tid = next_tid; next_tid += 1
            df.at[idx, "track_id"] = tid
            df.at[idx, "step_idx"] = 0
            cur_lat, cur_lon = float(row["lat"]), float(row["lon"])
            hcur = h
            step = 1
            while True:
                best = None
                # search next hours up to max_gap_hours
                for gap in range(1, max_gap_hours + 2):  # 1..max_gap+1 hours ahead
                    htry = hcur + pd.Timedelta(hours=gap)
                    cand = by_hour.get(htry)
                    if cand is None or cand.empty:
                        continue
                    # only unassigned and strictly forward in time
                    cand = cand[cand["index"].map(lambda j: df.at[int(j), "track_id"] == -1)]
                    if cand.empty:
                        continue
                    # distances
                    dists = hav_km(cur_lat, cur_lon, cand["lat"].to_numpy(), cand["lon"].to_numpy())
                    # pick nearest
                    j = int(np.argmin(dists))
                    dmin = float(dists[j])
                    if dmin <= link_radius_km:
                        row2 = cand.iloc[j]
                        best = {
                            "idx2": int(row2["index"]),
                            "gap_hours": gap,
                            "d_km": dmin,
                            "time": row2["time"],
                            "lat": float(row2["lat"]),
                            "lon": float(row2["lon"]),
                        }
                        break
                if best is None:
                    break
                # link
                idx2 = best["idx2"]
                df.at[idx2, "track_id"] = tid
                df.at[idx2, "step_idx"] = step
                df.at[idx2, "gap_count"] = best["gap_hours"] - 1
                cur_lat, cur_lon = best["lat"], best["lon"]
                hcur = best["time"]
                step += 1
    return df


def link_tracks_fast(
    seeds: pd.DataFrame,
    link_radius_km: float = 75.0,
    max_gap_hours: int = 1,
    log_every: int = 500,
) -> pd.DataFrame:
    """
    Faster greedy linker using per-hour BallTree queries to avoid O(n^2) scans.
    Preserves one-to-one matching and gap allowances up to max_gap_hours+1.
    """
    df = seeds.sort_values("time").reset_index(drop=True).copy()
    df["track_id"] = -1
    df["step_idx"] = -1
    df["gap_count"] = 0

    by_hour = df.groupby("time", sort=True).indices
    hours = sorted(by_hour.keys())
    next_tid = 1
    active: Dict[int, Dict[str, object]] = {}
    allowed_gap = max_gap_hours + 1

    for h_idx, h in enumerate(hours, start=1):
        idxs = np.asarray(by_hour[h], dtype=np.int64)
        if idxs.size == 0:
            continue

        # prune stale tracks
        stale = []
        for tid, st in active.items():
            gap_h = (pd.Timestamp(h) - pd.Timestamp(st["time"])).total_seconds() / 3600.0
            if gap_h > allowed_gap:
                stale.append(tid)
        for tid in stale:
            active.pop(tid, None)

        cur_lat = df.loc[idxs, "lat"].to_numpy(dtype=float)
        cur_lon = df.loc[idxs, "lon"].to_numpy(dtype=float)
        assigned_curr: set[int] = set()
        matched_tracks: set[int] = set()

        if active and _HAVE_BALLTREE:
            coords = np.c_[np.radians(cur_lat), np.radians(cur_lon)]
            tree = BallTree(coords, metric="haversine")

            active_items = list(active.items())
            active_tids = np.array([tid for tid, _ in active_items], dtype=np.int64)
            active_lat = np.array([st["lat"] for _, st in active_items], dtype=float)
            active_lon = np.array([st["lon"] for _, st in active_items], dtype=float)
            active_step = np.array([st["step_idx"] for _, st in active_items], dtype=np.int64)
            active_gap = np.array(
                [
                    (pd.Timestamp(h) - pd.Timestamp(st["time"])).total_seconds() / 3600.0
                    for _, st in active_items
                ],
                dtype=float,
            )

            dist, ind = tree.query(np.c_[np.radians(active_lat), np.radians(active_lon)], k=1)
            d_km = dist[:, 0] * 6371.0

            candidates = []
            for a_idx, dk in enumerate(d_km):
                if dk <= link_radius_km:
                    candidates.append((dk, a_idx, int(ind[a_idx][0])))
            candidates.sort(key=lambda x: x[0])

            for dk, a_idx, cur_local in candidates:
                tid = int(active_tids[a_idx])
                if tid in matched_tracks or cur_local in assigned_curr:
                    continue
                cur_global = int(idxs[cur_local])
                df.at[cur_global, "track_id"] = tid
                df.at[cur_global, "step_idx"] = int(active_step[a_idx]) + 1
                gap_h = int(round(active_gap[a_idx]))
                df.at[cur_global, "gap_count"] = max(gap_h - 1, 0)
                active[tid] = {
                    "time": h,
                    "lat": float(cur_lat[cur_local]),
                    "lon": float(cur_lon[cur_local]),
                    "step_idx": int(active_step[a_idx]) + 1,
                }
                matched_tracks.add(tid)
                assigned_curr.add(cur_local)
        elif active and not _HAVE_BALLTREE:
            print("[warn] fast linker requires sklearn BallTree; falling back to slow linker.")
            return link_tracks_slow(seeds, link_radius_km=link_radius_km, max_gap_hours=max_gap_hours)

        # start new tracks for unassigned current seeds
        for local_idx, cur_global in enumerate(idxs):
            if local_idx in assigned_curr:
                continue
            tid = next_tid
            next_tid += 1
            df.at[cur_global, "track_id"] = tid
            df.at[cur_global, "step_idx"] = 0
            df.at[cur_global, "gap_count"] = 0
            active[tid] = {
                "time": h,
                "lat": float(cur_lat[local_idx]),
                "lon": float(cur_lon[local_idx]),
                "step_idx": 0,
            }

        if log_every and (h_idx % log_every == 0):
            print(f"[info] linking pass: {h_idx} hours processed active_tracks={len(active):,}", flush=True)

    return df


def link_tracks(seeds: pd.DataFrame,
                link_radius_km: float = 75.0,
                max_gap_hours: int = 1) -> pd.DataFrame:
    """
    Choose a linker based on data size and availability.
    """
    if len(seeds) >= 1_000_000 and _HAVE_BALLTREE:
        print(f"[info] using fast linker for {len(seeds):,} seed rows (gap<= {max_gap_hours}h).", flush=True)
        return link_tracks_fast(seeds, link_radius_km=link_radius_km, max_gap_hours=max_gap_hours)
    return link_tracks_slow(seeds, link_radius_km=link_radius_km, max_gap_hours=max_gap_hours)

# ------------------ IBTrACS loading + KD-tree matching ------------------

def _ecef_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Convert lat/lon (deg) to ECEF XYZ on sphere (km)."""
    R = 6371.0
    lat = np.radians(lat_deg); lon = np.radians(lon_deg)
    X = np.c_[R*np.cos(lat)*np.cos(lon),
              R*np.cos(lat)*np.sin(lon),
              R*np.sin(lat)]
    return X

def load_ibtracs(path: str, normalize_lon: str = "-180..180", time_fmt: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    ib = read_any(path)
    # time
    tcol = None
    for c in ["iso_time","ISO_TIME","time","datetime","date_time","ObsTime","obs_time"]:
        if c in ib.columns:
            tcol = c; break
    if tcol is None:
        tcol = ib.columns[0]
    ib["_t_"] = to_utc_naive(ib[tcol], time_fmt)

    # lat/lon/vmax
    latc = "latitude" if "latitude" in ib.columns else ("lat" if "lat" in ib.columns else "LAT")
    lonc = "longitude" if "longitude" in ib.columns else ("lon" if "lon" in ib.columns else "LON")
    vmaxc = "wind_wmo" if "wind_wmo" in ib.columns else ("USA_WIND" if "USA_WIND" in ib.columns else
             ("WMO_WIND" if "WMO_WIND" in ib.columns else "vmax"))

    ib["lat"] = pd.to_numeric(ib[latc], errors="coerce")
    ib["lon"] = norm_lon(pd.to_numeric(ib[lonc], errors="coerce"), normalize_lon)
    ib["vmax"] = pd.to_numeric(ib[vmaxc], errors="coerce")

    # storm id
    sid = None
    for c in ["sid","SID","usa_atcf_id","identifier","serial_num","storm_id","ID","num","NUMBER"]:
        if c in ib.columns:
            sid = c; break
    if sid is None:
        namec = "name" if "name" in ib.columns else ("NAME" if "NAME" in ib.columns else None)
        seas  = "season" if "season" in ib.columns else ("SEASON" if "SEASON" in ib.columns else None)
        num   = "number" if "number" in ib.columns else ("NUMBER" if "NUMBER" in ib.columns else None)
        if namec and seas and num:
            ib["_sid_"] = ib[namec].astype(str)+"_"+ib[seas].astype(str)+"_"+ib[num].astype(str)
        else:
            ib["_sid_"] = "storm"
        sid = "_sid_"

    ib = ib.dropna(subset=["_t_","lat","lon"]).reset_index(drop=True)
    return ib, sid

def _first_TS_time_and_vmax(ib: pd.DataFrame, sid: str) -> pd.DataFrame:
    # precompute per-storm first time reaching 34 kt and lifetime max vmax
    recs = []
    for stid, g in ib.groupby(sid, sort=False):
        g = g.sort_values("_t_")
        ts = g.loc[g["vmax"] >= 34.0, "_t_"]
        ts_t = ts.iloc[0] if len(ts) else pd.NaT
        vmax_max = float(g["vmax"].max()) if len(g) else float("nan")
        recs.append((stid, ts_t, vmax_max))
    return pd.DataFrame(recs, columns=["storm_id","t_TS","vmax_max_kt"])

def label_tracks_kdtree(track_pts: pd.DataFrame,
                        ib: pd.DataFrame,
                        sid: str,
                        storm_radius_km: float = 150.0,
                        lookahead_hours: int = 120) -> pd.DataFrame:
    """
    Hour-bucketed KD-trees of storm points in ECEF. Falls back to O(N*M) if sklearn is absent.
    """
    # First-TS and vmax summary
    st_df = _first_TS_time_and_vmax(ib, sid)

    # Build hourly trees
    ib["_hour"] = ib["_t_"].dt.floor("h")
    trees: Dict[pd.Timestamp, KDTree] = {}
    per_hour: Dict[pd.Timestamp, pd.DataFrame] = {}
    if _HAVE_SK:
        for h, g in ib.groupby("_hour", sort=True):
            X = _ecef_xyz(g["lat"].to_numpy(), g["lon"].to_numpy())
            if len(X):
                trees[h] = KDTree(X)
                per_hour[h] = g.reset_index(drop=True)

    out = []
    for tid, g in track_pts.groupby("track_id", sort=False):
        g = g.sort_values("time")
        t0 = g["time"].iloc[0]
        t_end = t0 + pd.Timedelta(hours=lookahead_hours)
        matched = None
        best_d = 1e18

        if _HAVE_SK and trees:
            # Query per-hour trees in window
            P = _ecef_xyz(g["lat"].to_numpy(), g["lon"].to_numpy())
            for th in pd.date_range(t0.floor("h"), t_end.floor("h"), freq="h"):
                T = trees.get(th)
                G = per_hour.get(th)
                if T is None or G is None or G.empty:
                    continue
                dist, idx = T.query(P, k=1, return_distance=True)
                dmin = float(np.min(dist))
                if dmin < best_d:
                    best_d = dmin
                    flat_idx = np.argmin(dist)
                    j = int(np.asarray(idx).reshape(-1)[flat_idx])
                    matched = G.iloc[j][sid]
        else:
            # Fallback: brute force windowed search
            ibw = ib.loc[(ib["_t_"] >= t0) & (ib["_t_"] <= t_end)]
            for _, tp in g.iterrows():
                d = hav_km(tp["lat"], tp["lon"], ibw["lat"].to_numpy(), ibw["lon"].to_numpy())
                if len(d):
                    j = int(np.argmin(d))
                    dmin = float(d[j])
                    if dmin < best_d:
                        best_d = dmin
                        matched = ibw.iloc[j][sid]

        if matched is not None and best_d <= storm_radius_km:
            row = st_df.loc[st_df["storm_id"] == matched]
            t_TS = row["t_TS"].iloc[0] if len(row) else pd.NaT
            vmaxM = row["vmax_max_kt"].iloc[0] if len(row) else float("nan")
            lead_TS_h = (t_TS - t0).total_seconds()/3600.0 if pd.notna(t_TS) else float("nan")
            cat = (0 if not np.isfinite(vmaxM) else
                   (1 if vmaxM < 34 else
                    (2 if vmaxM < 50 else
                     (3 if vmaxM < 64 else
                      (4 if vmaxM < 83 else
                       (5 if vmaxM < 96 else
                        (6 if vmaxM < 113 else 7)))))))
            out.append(dict(track_id=tid, matched=True, storm_id=str(matched),
                            lead_to_TS_h=float(lead_TS_h), vmax_max_kt=float(vmaxM), cat_max=int(cat)))
        else:
            out.append(dict(track_id=tid, matched=False, storm_id="",
                            lead_to_TS_h=float("nan"), vmax_max_kt=float("nan"), cat_max=0))
    return pd.DataFrame(out)

# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser(description="Build proto-tracks from hourly seeds and label outcomes vs IBTrACS.")
    ap.add_argument("--seeds", required=True, help="CSV/Parquet with hourly seeds (time/lat/lon[, prob...]).")
    ap.add_argument("--features", default=None, help="Optional CSV/Parquet features with CAPE/CIN/T2M to join.")
    ap.add_argument("--ibtracs", required=True, help="IBTrACS CSV/Parquet (v04 list or similar).")
    ap.add_argument("--time-format", default=None, help="Optional strptime format for non-standard time columns.")
    ap.add_argument("--normalize-lon", choices=["none","-180..180","0..360"], default="-180..180", type=str)
    ap.add_argument("--aoi", default=None, help='Optional AOI "latN,lonW,latS,lonE" after lon normalization.')
    ap.add_argument("--link-radius-km", type=float, default=75.0)
    ap.add_argument("--max-gap-hours", type=int, default=1)
    ap.add_argument("--storm-radius-km", type=float, default=150.0)
    ap.add_argument("--structure-radius-km", type=float, default=100.0,
                    help="Neighbour-count radius per hour (0 to skip; uses haversine KDTree when available).")
    ap.add_argument("--lookahead-hours", type=int, default=120)
    ap.add_argument("--min-track-hours", type=int, default=2)
    ap.add_argument("--out-dir", default="results/seedmaps")
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--write-parquet", action="store_true", help="Also write Parquet copies.")
    ap.add_argument(
        "--points-format",
        choices=["csv", "parquet", "both"],
        default="csv",
        help="Format for track_points output (tracks/conversion follow CSV + optional parquet).",
    )
    ap.add_argument("--chunk-rows", type=int, default=0, help="Optional chunk size for CSV inputs (0=off).")
    ap.add_argument("--max-rows", type=int, default=None, help="Optional cap on seed rows (sample if larger).")
    ap.add_argument("--flag-col", default=None,
                    help="Optional seed flag column (e.g. any_alert); keep only rows where it is > 0 "
                         "so outcomes/conversion reflect fired alerts, not the whole union.")
    # preprocess normalize-lon to handle tokens like "-180..180"
    argv = []
    skip = False
    raw = sys.argv[1:]
    for i, tok in enumerate(raw):
        if skip:
            skip = False
            continue
        if tok == "--normalize-lon" and i + 1 < len(raw):
            argv.append(f"--normalize-lon={raw[i+1].strip()}")
            skip = True
        elif tok.startswith("--normalize-lon="):
            lhs, rhs = tok.split("=", 1)
            argv.append(f"{lhs}={rhs.strip()}")
        else:
            argv.append(tok)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load seeds
    s = read_csv_chunked(args.seeds, args.chunk_rows, args.max_rows)
    if args.flag_col:
        if args.flag_col in s.columns:
            n0 = len(s)
            s = s[pd.to_numeric(s[args.flag_col], errors="coerce").fillna(0) > 0]
            print(f"[info] flag gate {args.flag_col}>0: {n0:,} -> {len(s):,} seed rows")
        else:
            print(f"[warn] --flag-col '{args.flag_col}' not in seeds; no gate applied.",
                  file=sys.stderr)
    if args.max_rows and len(s) > args.max_rows:
        s = s.sample(n=int(args.max_rows), random_state=42)
        print(f"[info] seeds sampled to {len(s):,} rows (max_rows={args.max_rows})")
    print(f"[info] loaded seeds rows={len(s):,}")
    tcol = pick_time_col(s)
    s["time"] = to_utc_naive(s[tcol], args.time_format)
    s["lat"]  = pd.to_numeric(s["lat"], errors="coerce")
    s["lon"]  = norm_lon(pd.to_numeric(s["lon"], errors="coerce"), args.normalize_lon)
    pcol = pick_prob_col(s)
    s["prob"] = pd.to_numeric(s[pcol], errors="coerce") if pcol else np.nan
    s = s.dropna(subset=["time","lat","lon"]).reset_index(drop=True)

    # AOI crop (optional) before structure/linking
    aoi = parse_area(args.aoi)
    if aoi:
        s = crop_area(s, aoi)

    # per-hour simple structure
    if args.structure_radius_km <= 0:
        s["nbr_100km"] = 0.0
    else:
        nbr = np.zeros(len(s), dtype=np.float32)
        groups = s.groupby("time", sort=True).indices
        for idx, (ts, idxs) in enumerate(groups.items(), start=1):
            latv = s.loc[idxs, "lat"].to_numpy()
            lonv = s.loc[idxs, "lon"].to_numpy()
            nbr[idxs] = _structure_counts(latv, lonv, args.structure_radius_km)
            if idx % 500 == 0:
                print(f"[info] structure pass: {idx} hours processed", flush=True)
        s["nbr_100km"] = nbr

    # ---- optional: join CAPE/CIN/T2M
    if args.features:
        f = read_csv_chunked(args.features, args.chunk_rows)
        print(f"[info] loaded features rows={len(f):,}")
        # normalize time/coords
        if "time" not in f.columns:
            for c in ["valid_time","datetime","time_h"]:
                if c in f.columns:
                    f = f.rename(columns={c:"time"}); break
        f["time"] = to_utc_naive(f["time"], args.time_format)
        f["lat"]  = pd.to_numeric(f["lat"], errors="coerce")
        f["lon"]  = norm_lon(pd.to_numeric(f["lon"], errors="coerce"), args.normalize_lon)
        f = f.dropna(subset=["time","lat","lon"]).reset_index(drop=True)
        if aoi:
            f = crop_area(f, aoi)
        cape_cols = pick_cape_cols(f)
        keep = ["time","lat","lon"] + list(cape_cols.values())
        f = f[keep].copy()
        left_df = s
        s = s.merge(f, on=["time","lat","lon"], how="left", validate="m:1")
        left_dupe = int(left_df.duplicated(subset=["time", "lat", "lon"]).sum())
        right_dupe = int(f.duplicated(subset=["time", "lat", "lon"]).sum())
        unmatched = join_audit.estimate_unmatched_keys(left_df, f, ["time", "lat", "lon"])
        entry = join_audit.build_entry(
            step="seeds.proto-outcomes.features-merge",
            keys=["time", "lat", "lon"],
            join_type="left",
            left_rows=len(left_df),
            right_rows=len(f),
            out_rows=len(s),
            left_dupe_keys=left_dupe,
            right_dupe_keys=right_dupe,
            left_key_count=unmatched.get("left_key_count"),
            right_key_count=unmatched.get("right_key_count"),
            left_unmatched_keys=unmatched.get("left_unmatched_keys"),
            right_unmatched_keys=unmatched.get("right_unmatched_keys"),
            unmatched_sampled=unmatched.get("unmatched_sampled"),
            extra={"features_path": str(args.features)},
        )
        join_audit.append_entry(join_audit.default_path(), entry)
        # unify names if present
        rev = {v:k for k,v in cape_cols.items()}
        s = s.rename(columns=rev)

    # ---- link hour-to-hour into proto-tracks
    keep_cols = ["time","lat","lon","prob"] + [c for c in ["nbr_100km","cape","cin","t2m"] if c in s.columns]
    linked = link_tracks(s[keep_cols].copy(),
                         link_radius_km=args.link_radius_km,
                         max_gap_hours=args.max_gap_hours)
    print(f"[info] linked proto-tracks: rows={len(linked):,} tracks={linked['track_id'].nunique():,}")

    # ---- points table + rank index within track
    pts = linked.copy()
    pts["hour_idx"] = pts.groupby("track_id")["time"].rank(method="first").astype(int) - 1

    # filter short tracks
    counts = pts.groupby("track_id").size().rename("n_hours").reset_index()
    good_ids = counts.loc[counts["n_hours"] >= int(args.min_track_hours), "track_id"].tolist()
    pts = pts[pts["track_id"].isin(good_ids)].reset_index(drop=True)
    print(f"[info] filtered tracks to >= {args.min_track_hours}h: tracks={len(good_ids):,} rows={len(pts):,}")

    # ---- per-track summary
    def qnan(x, q):
        x = pd.to_numeric(x, errors="coerce")
        return float(np.nanquantile(x, q)) if len(x) else float("nan")

    agg = (pts.groupby("track_id")
             .agg(
                 start_time=("time","min"),
                 end_time=("time","max"),
                 duration_h=("time", lambda x: (x.max()-x.min()).total_seconds()/3600.0 + 1.0),
                 start_lat=("lat","first"),
                 start_lon=("lon","first"),
                 end_lat=("lat","last"),
                 end_lon=("lon","last"),
                 mean_prob=("prob","mean"),
                 max_prob=("prob","max"),
                 mean_nbr=("nbr_100km","mean") if "nbr_100km" in pts.columns else ("prob","size"),
                 mean_cape=("cape","mean") if "cape" in pts.columns else ("prob","size"),
                 mean_cin =("cin","mean")  if "cin"  in pts.columns else ("prob","size"),
                 mean_t2m=("t2m","mean")  if "t2m"  in pts.columns else ("prob","size"),
                 p50_prob=("prob", lambda x: qnan(x,0.5)),
                 p90_prob=("prob", lambda x: qnan(x,0.9)),
             ).reset_index())

    # ---- load IBTrACS and match outcomes
    ib, sid = load_ibtracs(args.ibtracs, normalize_lon=args.normalize_lon, time_fmt=args.time_format)
    if aoi:
        ib = crop_area(ib.rename(columns={"_t_":"__t"}), aoi).rename(columns={"__t":"_t_"})
    outcomes = label_tracks_kdtree(pts, ib, sid,
                                   storm_radius_km=args.storm_radius_km,
                                   lookahead_hours=args.lookahead_hours)

    left_tracks = agg
    tracks = agg.merge(outcomes, on="track_id", how="left")
    left_dupe = int(left_tracks.duplicated(subset=["track_id"]).sum())
    right_dupe = int(outcomes.duplicated(subset=["track_id"]).sum())
    unmatched = join_audit.estimate_unmatched_keys(left_tracks, outcomes, ["track_id"])
    entry = join_audit.build_entry(
        step="seeds.proto-outcomes.track-merge",
        keys=["track_id"],
        join_type="left",
        left_rows=len(left_tracks),
        right_rows=len(outcomes),
        out_rows=len(tracks),
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

    # ---- aggregate conversion & conditional severity (correct conditioning)
    conv_rows = []
    for H in [24, 48, 72, 120]:
        within = tracks["lead_to_TS_h"].le(H)
        denom_mask = within.notna()
        conv_rate = float(within[denom_mask].mean()) if denom_mask.any() else float("nan")

        matched_within = within.fillna(False)
        if matched_within.any():
            cat1p = (tracks["cat_max"] >= 4) & matched_within
            p_cat1p = float(cat1p.sum() / matched_within.sum())
        else:
            p_cat1p = float("nan")

        conv_rows.append({
            "horizon_h": H,
            "conv_to_TS_rate": conv_rate,
            "P_Cat1plus_given_match": p_cat1p
        })
    conv_df = pd.DataFrame(conv_rows)

    # ---- write outputs
    base = f"{args.run_name}"
    out_pts_base = Path(args.out_dir) / f"{base}_track_points"
    out_tr_base  = Path(args.out_dir) / f"{base}_tracks"
    out_conv_base= Path(args.out_dir) / f"{base}_conversion_rates"

    want_csv = args.points_format in ("csv", "both")
    want_parquet_pts = args.points_format in ("parquet", "both") or args.write_parquet

    if want_csv:
        out_pts_csv = out_pts_base.with_suffix(".csv")
        pts.to_csv(out_pts_csv, index=False, date_format="%Y-%m-%d %H:00:00")
        print(f"[write] {out_pts_csv} rows={len(pts)}")
    if want_parquet_pts:
        out_pts_parq = out_pts_base.with_suffix(".parquet")
        pts.to_parquet(out_pts_parq, index=False)
        print(f"[write] {out_pts_parq} rows={len(pts)}")

    out_tr_csv = out_tr_base.with_suffix(".csv")
    tracks.to_csv(out_tr_csv, index=False, date_format="%Y-%m-%d %H:00:00")
    out_conv_csv = out_conv_base.with_suffix(".csv")
    conv_df.to_csv(out_conv_csv, index=False)
    print(f"[write] {out_tr_csv} rows={len(tracks)}")
    print(f"[write] {out_conv_csv} rows={len(conv_df)}")

    if args.write_parquet:
        tracks.to_parquet(str(out_tr_csv) + ".parquet", index=False)
        conv_df.to_parquet(str(out_conv_csv) + ".parquet", index=False)

    print(f"[write] {out_tr_csv} rows={len(tracks)}")
    print(f"[write] {out_conv_csv} rows={len(conv_df)}")

if __name__ == "__main__":
    main()
