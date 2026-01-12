#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
match_objects_to_tracks.py

Agent: link object components to storm tracks and derive motion/directionality metrics.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
from utils import join_audit


def _is_parquet(path: str | Path) -> bool:
    low = str(path).lower()
    return low.endswith((".parquet", ".parq", ".pq", ".pqt"))


def _read_any(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if _is_parquet(p):
        return pd.read_parquet(p)
    return pd.read_csv(p, low_memory=False)


def _norm_lon(series: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if mode == "none":
        return x
    if mode == "0..360":
        return (x % 360 + 360) % 360
    return ((x + 180) % 360) - 180


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2 * r * math.asin(math.sqrt(max(a, 0.0)))


def _bearing_deg(lat1, lon1, lat2, lon2) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _angle_diff_deg(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


def _bearing_from_uv(u: float, v: float) -> float:
    if not np.isfinite(u) or not np.isfinite(v):
        return float("nan")
    return (math.degrees(math.atan2(u, v)) + 360.0) % 360.0


def _speed_kmh_from_uv(u: float, v: float) -> float:
    if not np.isfinite(u) or not np.isfinite(v):
        return float("nan")
    return float(math.hypot(u, v) * 3.6)


def _circular_std_deg(angles_deg: np.ndarray) -> float:
    angles = np.asarray(angles_deg, dtype=float)
    angles = angles[np.isfinite(angles)]
    if angles.size < 2:
        return float("nan")
    rad = np.deg2rad(angles)
    sin_mean = float(np.nanmean(np.sin(rad)))
    cos_mean = float(np.nanmean(np.cos(rad)))
    r = math.hypot(sin_mean, cos_mean)
    if not np.isfinite(r) or r <= 0:
        return float("nan")
    std = math.sqrt(max(0.0, -2.0 * math.log(max(r, 1e-12))))
    return float(np.degrees(std))


def _pick_track_cols(df: pd.DataFrame, overrides: Dict[str, Optional[str]]) -> Dict[str, str]:
    def _first(cands: Iterable[str]) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
        return None

    time_col = overrides.get("time") or _first(["time", "obs_time", "datetime", "valid_time"])
    lat_col = overrides.get("lat") or _first(["lat", "latitude"])
    lon_col = overrides.get("lon") or _first(["lon", "longitude"])
    id_col = overrides.get("id") or _first(["storm_id", "sid", "name"])
    vmax_col = overrides.get("vmax") or _first(["vmax", "wind", "maxwind"])
    if not time_col or not lat_col or not lon_col:
        raise SystemExit("Tracks file missing required time/lat/lon columns.")
    return {"time": time_col, "lat": lat_col, "lon": lon_col, "id": id_col or "storm_id", "vmax": vmax_col or ""}


def _add_track_motion(tr: pd.DataFrame, id_col: str) -> pd.DataFrame:
    out = tr.copy()
    out["track_bearing_deg"] = np.nan
    out["track_speed_kmh"] = np.nan
    for _, g in out.groupby(id_col, sort=False):
        idx = g.sort_values("time").index.to_numpy()
        for i in range(len(idx) - 1):
            a = out.loc[idx[i]]
            b = out.loc[idx[i + 1]]
            dt_h = (b["time"] - a["time"]).total_seconds() / 3600.0
            if dt_h <= 0:
                continue
            dist = _haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
            out.loc[idx[i], "track_bearing_deg"] = _bearing_deg(a["lat"], a["lon"], b["lat"], b["lon"])
            out.loc[idx[i], "track_speed_kmh"] = dist / dt_h
    return out


def _select_score_col(df: pd.DataFrame, pref: Optional[str]) -> str:
    if pref and pref in df.columns:
        return pref
    for cand in ["obj_score_topk_mean", "obj_score_max", "obj_score_mean"]:
        if cand in df.columns:
            return cand
    raise SystemExit("No usable object score column found.")


def _add_flow_direction(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "obj_u10_mean" in out.columns and "obj_v10_mean" in out.columns:
        u = pd.to_numeric(out["obj_u10_mean"], errors="coerce")
        v = pd.to_numeric(out["obj_v10_mean"], errors="coerce")
        out["obj_flow_bearing_deg"] = [
            _bearing_from_uv(ui, vi) for ui, vi in zip(u.to_numpy(), v.to_numpy())
        ]
        out["obj_flow_speed_kmh"] = [
            _speed_kmh_from_uv(ui, vi) for ui, vi in zip(u.to_numpy(), v.to_numpy())
        ]
    else:
        out["obj_flow_bearing_deg"] = np.nan
        out["obj_flow_speed_kmh"] = np.nan
    return out


def _link_objects(
    df: pd.DataFrame,
    max_link_km: float,
    dist_scale_km: float,
    score_scale: float,
    heading_scale_deg: float,
    axis_col: str,
    score_col: str,
    time_col: str,
) -> pd.DataFrame:
    out = df.copy()
    out["obj_next_id"] = np.nan
    out["obj_motion_bearing_deg"] = np.nan
    out["obj_motion_speed_kmh"] = np.nan

    if time_col not in out.columns:
        raise SystemExit(f"[match] objects missing time column '{time_col}' for linking.")
    groups = {t: g for t, g in out.groupby(time_col, sort=True)}
    times = sorted(groups.keys())
    next_map: Dict[int, int] = {}

    for t in times:
        t_next = t + pd.Timedelta(hours=1)
        g = groups.get(t)
        g_next = groups.get(t_next)
        if g is None or g_next is None or g.empty or g_next.empty:
            continue
        lat_next = g_next["obj_centroid_lat"].to_numpy()
        lon_next = g_next["obj_centroid_lon"].to_numpy()
        score_next = g_next[score_col].to_numpy()

        for _, row in g.iterrows():
            lat0 = row["obj_centroid_lat"]
            lon0 = row["obj_centroid_lon"]
            if not np.isfinite(lat0) or not np.isfinite(lon0):
                continue
            dists = np.array([_haversine_km(lat0, lon0, la, lo) for la, lo in zip(lat_next, lon_next)], dtype=float)
            within = dists <= float(max_link_km)
            if not within.any():
                continue
            bearing_to = np.array([_bearing_deg(lat0, lon0, la, lo) for la, lo in zip(lat_next, lon_next)], dtype=float)
            axis = row.get(axis_col, np.nan)
            heading_mismatch = np.zeros_like(dists)
            if np.isfinite(axis):
                heading_mismatch = np.abs([_angle_diff_deg(axis, b) for b in bearing_to])

            score0 = row[score_col]
            score_diff = np.abs(score_next - score0)

            cost = (dists / max(dist_scale_km, 1e-6)) + (score_diff / max(score_scale, 1e-6)) + (heading_mismatch / max(heading_scale_deg, 1e-6))
            cost = np.where(within, cost, np.inf)
            j = int(np.nanargmin(cost))
            if not np.isfinite(cost[j]):
                continue
            next_id = int(g_next.iloc[j]["object_id"])
            next_map[int(row["object_id"])] = next_id

            dt_h = (t_next - t).total_seconds() / 3600.0
            out.loc[row.name, "obj_next_id"] = next_id
            out.loc[row.name, "obj_motion_bearing_deg"] = bearing_to[j]
            out.loc[row.name, "obj_motion_speed_kmh"] = dists[j] / max(dt_h, 1e-6)

    # Tracklet IDs based on links
    prev_map = {v: k for k, v in next_map.items()}
    track_id_map: Dict[int, int] = {}
    next_track_id = 1
    for _, row in out.sort_values(time_col).iterrows():
        obj_id = int(row["object_id"])
        prev = prev_map.get(obj_id)
        if prev is not None and prev in track_id_map:
            track_id_map[obj_id] = track_id_map[prev]
        else:
            track_id_map[obj_id] = next_track_id
            next_track_id += 1
    out["obj_track_id"] = out["object_id"].map(track_id_map)
    return out


def _add_directionality(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["obj_axis_align_motion"] = np.nan
    out["obj_corepull_align_motion"] = np.nan
    for idx, row in out.iterrows():
        motion = row.get("obj_motion_bearing_deg", np.nan)
        axis = row.get("obj_axis_bearing_deg", np.nan)
        if np.isfinite(motion) and np.isfinite(axis):
            diff = _angle_diff_deg(axis, motion)
            out.loc[idx, "obj_axis_align_motion"] = math.cos(math.radians(diff))
        core_lat = row.get("obj_core_lat", np.nan)
        core_lon = row.get("obj_core_lon", np.nan)
        cen_lat = row.get("obj_centroid_lat", np.nan)
        cen_lon = row.get("obj_centroid_lon", np.nan)
        if np.isfinite(motion) and np.isfinite(core_lat) and np.isfinite(core_lon) and np.isfinite(cen_lat) and np.isfinite(cen_lon):
            core_bearing = _bearing_deg(cen_lat, cen_lon, core_lat, core_lon)
            diff = _angle_diff_deg(core_bearing, motion)
            out.loc[idx, "obj_corepull_align_motion"] = math.cos(math.radians(diff))
    return out


def _add_motion_uncertainty(df: pd.DataFrame, hours: float, time_col: str) -> pd.DataFrame:
    out = df.copy()
    out["obj_motion_uncertainty_deg"] = np.nan
    if "obj_track_id" not in out.columns or "obj_motion_bearing_deg" not in out.columns:
        return out
    if time_col not in out.columns:
        return out
    window = pd.Timedelta(hours=float(hours))
    for track_id, g in out.groupby("obj_track_id", sort=False):
        g = g.sort_values(time_col)
        times = g[time_col].to_numpy()
        bearings = g["obj_motion_bearing_deg"].to_numpy(dtype=float)
        idxs = g.index.to_numpy()
        for i, idx in enumerate(idxs):
            t0 = times[i]
            mask = (times >= (t0 - window)) & (times <= t0)
            out.loc[idx, "obj_motion_uncertainty_deg"] = _circular_std_deg(bearings[mask])
    return out


def _stable_hash_order(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return np.arange(len(df), dtype=np.int64)
    hashed = hash_pandas_object(df[cols], index=False, hash_key="matchq")
    return np.argsort(hashed.to_numpy(dtype=np.uint64), kind="mergesort")


def _nms_by_distance(df: pd.DataFrame, min_sep_km: float, top_k: int) -> pd.DataFrame:
    if df.empty:
        return df
    kept = []
    cap = int(top_k) if top_k is not None else 0
    for _, row in df.iterrows():
        lat = row["obj_centroid_lat"]
        lon = row["obj_centroid_lon"]
        if not np.isfinite(lat) or not np.isfinite(lon):
            continue
        if kept:
            dmin = min(_haversine_km(lat, lon, k["obj_centroid_lat"], k["obj_centroid_lon"]) for k in kept)
            if dmin < float(min_sep_km):
                continue
        kept.append(row)
        if cap > 0 and len(kept) >= cap:
            break
    return pd.DataFrame(kept)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Match per-hour objects to tracks and compute motion/directionality metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--objects", required=True, help="objects_by_hour table (CSV/Parquet).")
    ap.add_argument("--tracks", required=True, help="IBTrACS subset (CSV/Parquet).")
    ap.add_argument("--out", default="results/matches/storm_object_matches.parquet", help="Matched pairs output.")
    ap.add_argument("--objects-out", default=None, help="Optional objects table with motion columns.")
    ap.add_argument("--tracks-out", default=None, help="Optional tracks table with motion columns.")
    ap.add_argument("--normalize-lon", choices=["none", "-180..180", "0..360"], default="-180..180")
    ap.add_argument("--time-col", default="time")
    ap.add_argument("--score-col", default=None, help="Object score column (defaults to obj_score_topk_mean).")
    ap.add_argument("--match-radius-km", type=float, default=150.0)
    ap.add_argument("--time-tol-hours", type=float, default=6.0)
    ap.add_argument("--sigma-dist-km", type=float, default=None)
    ap.add_argument("--sigma-time-h", type=float, default=None)
    ap.add_argument("--top-n", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=20, help="Max candidates per storm-hour after scoring + de-dup.")
    ap.add_argument("--min-sep-km", type=float, default=75.0, help="Minimum separation for NMS de-dup.")
    ap.add_argument("--score-dist-km", type=float, default=None, help="Distance scale d0 for match scoring.")
    ap.add_argument("--w-prob", type=float, default=1.0, help="Weight for object score (P).")
    ap.add_argument("--w-dist", type=float, default=1.0, help="Weight for distance decay term.")
    ap.add_argument("--w-compact", type=float, default=0.5, help="Weight for compactness term.")
    ap.add_argument("--w-persist", type=float, default=0.5, help="Weight for temporal persistence term.")
    ap.add_argument("--w-area", type=float, default=0.0, help="Weight for area penalty term.")
    ap.add_argument("--area-scale-cells", type=float, default=100.0, help="Scale for area penalty (cells).")
    ap.add_argument("--persist-scale-hours", type=float, default=6.0, help="Scale for persistence term (hours).")
    ap.add_argument("--adaptive-quantile", type=float, default=0.995, help="Per-hour score quantile for gating.")
    ap.add_argument("--adaptive-base-threshold", type=float, default=None, help="Base score threshold for gating.")
    ap.add_argument("--link-max-km", type=float, default=120.0)
    ap.add_argument("--link-dist-scale-km", type=float, default=60.0)
    ap.add_argument("--link-score-scale", type=float, default=0.2)
    ap.add_argument("--link-heading-scale-deg", type=float, default=45.0)
    ap.add_argument("--motion-uncertainty-hours", type=float, default=6.0,
                    help="Window (hours) for motion bearing uncertainty.")
    ap.add_argument("--track-time-col", default=None)
    ap.add_argument("--track-lat-col", default=None)
    ap.add_argument("--track-lon-col", default=None)
    ap.add_argument("--track-id-col", default=None)
    ap.add_argument("--track-vmax-col", default=None)
    ap.add_argument("--chunk-rows", dest="chunk_rows", type=int, default=None, help="Ignored; accepted for pipeline compatibility.")
    ap.add_argument("--chunksize", dest="chunk_rows", type=int, default=None, help="Alias for --chunk-rows (ignored).")
    ap.add_argument("--parquet-rows", dest="parquet_rows", type=int, default=None, help="Ignored; accepted for pipeline compatibility.")
    args = ap.parse_args()

    obj = _read_any(args.objects)
    if obj.empty:
        raise SystemExit("[match] objects table is empty.")
    if args.time_col not in obj.columns:
        raise SystemExit(f"[match] objects table missing time column '{args.time_col}'.")
    obj = obj.copy()
    obj[args.time_col] = pd.to_datetime(obj[args.time_col], utc=True, errors="coerce").dt.tz_localize(None)
    obj = obj.dropna(subset=[args.time_col, "obj_centroid_lat", "obj_centroid_lon"]).reset_index(drop=True)
    obj["obj_centroid_lat"] = pd.to_numeric(obj["obj_centroid_lat"], errors="coerce")
    obj["obj_centroid_lon"] = _norm_lon(obj["obj_centroid_lon"], args.normalize_lon)

    score_col = _select_score_col(obj, args.score_col)
    obj_score = pd.to_numeric(obj[score_col], errors="coerce")
    obj["obj_score"] = obj_score
    if "obj_score_topk_mean" in obj.columns and "obj_score_mean" in obj.columns:
        mean = pd.to_numeric(obj["obj_score_mean"], errors="coerce")
        topk = pd.to_numeric(obj["obj_score_topk_mean"], errors="coerce")
        obj["obj_compactness"] = np.where(mean > 0, topk / mean, np.nan)
    else:
        obj["obj_compactness"] = np.nan

    # Object propagation (hour-to-hour)
    axis_col = "obj_axis_bearing_deg" if "obj_axis_bearing_deg" in obj.columns else ""
    obj = _link_objects(
        obj,
        max_link_km=args.link_max_km,
        dist_scale_km=args.link_dist_scale_km,
        score_scale=args.link_score_scale,
        heading_scale_deg=args.link_heading_scale_deg,
        axis_col=axis_col,
        score_col="obj_score",
        time_col=args.time_col,
    )
    obj = _add_directionality(obj)
    obj = _add_flow_direction(obj)
    obj = _add_motion_uncertainty(obj, args.motion_uncertainty_hours, args.time_col)
    if "obj_flow_bearing_deg" in obj.columns and obj["obj_flow_bearing_deg"].notna().sum() == 0:
        print("[match] flow-direction columns present but empty (u10/v10 likely missing).")

    if "obj_track_len_h" in obj.columns:
        obj["obj_track_len_h"] = pd.to_numeric(obj["obj_track_len_h"], errors="coerce")
    elif "obj_track_id" in obj.columns and args.time_col in obj.columns:
        track_len = obj.groupby("obj_track_id")[args.time_col].nunique().rename("obj_track_len_h")
        left_df = obj
        track_len_df = track_len.reset_index()
        obj = obj.merge(track_len, left_on="obj_track_id", right_index=True, how="left")
        left_dupe = int(left_df.duplicated(subset=["obj_track_id"]).sum())
        right_dupe = int(track_len_df.duplicated(subset=["obj_track_id"]).sum())
        unmatched = join_audit.estimate_unmatched_keys(left_df, track_len_df, ["obj_track_id"])
        entry = join_audit.build_entry(
            step="reports.match-objects.tracklen-merge",
            keys=["obj_track_id"],
            join_type="left",
            left_rows=len(left_df),
            right_rows=len(track_len_df),
            out_rows=len(obj),
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
        if "obj_track_len_h" not in obj.columns:
            for cand in ("obj_track_len_h_x", "obj_track_len_h_y"):
                if cand in obj.columns:
                    obj["obj_track_len_h"] = pd.to_numeric(obj[cand], errors="coerce")
                    break
    if "obj_track_len_h" not in obj.columns:
        obj["obj_track_len_h"] = np.nan
    persist_scale = max(float(args.persist_scale_hours), 1e-6)
    obj["obj_persist"] = np.clip(pd.to_numeric(obj["obj_track_len_h"], errors="coerce") / persist_scale, 0.0, 1.0)

    # Tracks
    tr = _read_any(args.tracks)
    if tr.empty:
        raise SystemExit("[match] tracks table is empty.")
    colmap = _pick_track_cols(
        tr,
        {
            "time": args.track_time_col,
            "lat": args.track_lat_col,
            "lon": args.track_lon_col,
            "id": args.track_id_col,
            "vmax": args.track_vmax_col,
        },
    )
    tr = tr.copy()
    tr = tr.rename(columns={colmap["time"]: "time", colmap["lat"]: "lat", colmap["lon"]: "lon"})
    if colmap["id"] in tr.columns:
        tr = tr.rename(columns={colmap["id"]: "storm_id"})
    else:
        tr["storm_id"] = "storm"
    if colmap["vmax"]:
        tr = tr.rename(columns={colmap["vmax"]: "vmax"})
    tr["time"] = pd.to_datetime(tr["time"], utc=True, errors="coerce").dt.tz_localize(None)
    tr["lat"] = pd.to_numeric(tr["lat"], errors="coerce")
    tr["lon"] = _norm_lon(tr["lon"], args.normalize_lon)
    tr = tr.dropna(subset=["time", "lat", "lon"]).reset_index(drop=True)

    tr = _add_track_motion(tr, "storm_id")

    # Agent: scored candidate matching (top-K + spatial de-dup) to reduce blobbing.
    # Matching (score + top-K + spatial de-dup)
    sigma_d = args.sigma_dist_km or (args.match_radius_km / 2.0)
    sigma_t = args.sigma_time_h or (args.time_tol_hours / 2.0)
    dist_scale = args.score_dist_km or sigma_d or (args.match_radius_km / 2.0)
    dt_tol = pd.Timedelta(hours=float(args.time_tol_hours))

    obj_groups = {t: g for t, g in obj.groupby(obj[args.time_col].dt.floor("h"))}
    adaptive_thr: Dict[pd.Timestamp, float] = {}
    base_thr = float(args.adaptive_base_threshold) if args.adaptive_base_threshold is not None else None
    if args.adaptive_quantile is not None and args.adaptive_quantile < 1.0:
        q = float(args.adaptive_quantile)
        for t_key, g in obj_groups.items():
            scores = pd.to_numeric(g["obj_score"], errors="coerce")
            scores = scores[np.isfinite(scores)]
            if scores.empty:
                continue
            thr = float(scores.quantile(q))
            if base_thr is not None:
                thr = max(base_thr, thr)
            adaptive_thr[t_key] = thr
    elif base_thr is not None:
        for t_key in obj_groups.keys():
            adaptive_thr[t_key] = base_thr

    rows: List[pd.DataFrame] = []
    for _, tr_row in tr.iterrows():
        t0 = tr_row["time"]
        hwin = pd.date_range(t0 - dt_tol, t0 + dt_tol, freq="h")
        cands: List[Dict[str, object]] = []
        for th in hwin:
            g = obj_groups.get(th)
            if g is None or g.empty:
                continue
            if th in adaptive_thr:
                thr = adaptive_thr[th]
                g = g.loc[pd.to_numeric(g["obj_score"], errors="coerce") >= thr]
                if g.empty:
                    continue
            for _, o in g.iterrows():
                d_km = _haversine_km(o["obj_centroid_lat"], o["obj_centroid_lon"], tr_row["lat"], tr_row["lon"])
                if d_km > float(args.match_radius_km):
                    continue
                dt_h = (o[args.time_col] - t0).total_seconds() / 3600.0
                score_prob = float(o["obj_score"]) if np.isfinite(o["obj_score"]) else 0.0
                compact = float(o.get("obj_compactness", 1.0)) if np.isfinite(o.get("obj_compactness", 1.0)) else 1.0
                compact = float(np.clip(compact, 0.0, 2.0))
                persist = float(o.get("obj_persist", 0.0)) if np.isfinite(o.get("obj_persist", 0.0)) else 0.0
                area_cells = float(o.get("obj_area_cells", np.nan)) if np.isfinite(o.get("obj_area_cells", np.nan)) else np.nan
                area_penalty = 0.0
                if np.isfinite(area_cells) and args.area_scale_cells and args.area_scale_cells > 0:
                    area_penalty = float(area_cells) / float(args.area_scale_cells)

                score_dist = math.exp(-d_km / max(dist_scale, 1e-6))
                match_score = (
                    args.w_prob * score_prob
                    + args.w_dist * score_dist
                    + args.w_compact * compact
                    + args.w_persist * persist
                    - args.w_area * area_penalty
                )

                sd = max(float(sigma_d), 1e-6)
                st = max(float(sigma_t), 1e-6)
                match_score_gauss = score_prob * math.exp(-(d_km ** 2) / (2 * sd ** 2)) * math.exp(-(dt_h ** 2) / (2 * st ** 2))
                match_score_gauss *= compact

                cands.append(
                    {
                        "storm_id": tr_row["storm_id"],
                        "track_time": t0,
                        "track_lat": tr_row["lat"],
                        "track_lon": tr_row["lon"],
                        "track_bearing_deg": tr_row.get("track_bearing_deg", np.nan),
                        "track_speed_kmh": tr_row.get("track_speed_kmh", np.nan),
                        "vmax": tr_row.get("vmax", np.nan),
                        "object_id": o["object_id"],
                        "obj_track_id": o.get("obj_track_id", np.nan),
                        "object_time": o[args.time_col],
                        "obj_centroid_lat": o["obj_centroid_lat"],
                        "obj_centroid_lon": o["obj_centroid_lon"],
                        "obj_score": o["obj_score"],
                        "obj_area_cells": o.get("obj_area_cells", np.nan),
                        "obj_motion_bearing_deg": o.get("obj_motion_bearing_deg", np.nan),
                        "obj_motion_speed_kmh": o.get("obj_motion_speed_kmh", np.nan),
                        "obj_motion_uncertainty_deg": o.get("obj_motion_uncertainty_deg", np.nan),
                        "obj_flow_bearing_deg": o.get("obj_flow_bearing_deg", np.nan),
                        "obj_flow_speed_kmh": o.get("obj_flow_speed_kmh", np.nan),
                        "obj_axis_bearing_deg": o.get("obj_axis_bearing_deg", np.nan),
                        "obj_axis_align_motion": o.get("obj_axis_align_motion", np.nan),
                        "obj_corepull_align_motion": o.get("obj_corepull_align_motion", np.nan),
                        "obj_persist": o.get("obj_persist", np.nan),
                        "obj_track_len_h": o.get("obj_track_len_h", np.nan),
                        "score_prob": score_prob,
                        "score_dist": score_dist,
                        "score_compact": compact,
                        "score_persist": persist,
                        "score_area_penalty": area_penalty,
                        "d_km": d_km,
                        "dt_hours": dt_h,
                        "match_score": match_score,
                        "match_score_gauss": match_score_gauss,
                    }
                )
        if not cands:
            continue
        cand_df = pd.DataFrame(cands)
        cand_df = cand_df.sort_values(["match_score"], ascending=False)
        # Break ties deterministically to avoid spatial banding
        tie_order = _stable_hash_order(cand_df, ["object_id", "object_time"])
        tie_rank = np.empty(len(cand_df), dtype=np.int64)
        tie_rank[tie_order] = np.arange(len(cand_df))
        cand_df["__tie__"] = tie_rank
        cand_df = cand_df.sort_values(["match_score", "__tie__"], ascending=[False, True])
        cand_df = cand_df.drop(columns=["__tie__"])
        cand_df = _nms_by_distance(cand_df, min_sep_km=args.min_sep_km, top_k=int(args.top_k))
        if cand_df.empty:
            continue
        cand_df = cand_df.sort_values("match_score", ascending=False).reset_index(drop=True)
        top_n = min(int(args.top_n), len(cand_df))
        cand_df["match_rank"] = np.arange(1, len(cand_df) + 1)
        cand_df["match_is_primary"] = cand_df["match_rank"] <= top_n
        rows.append(cand_df)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if _is_parquet(out_path):
        out.to_parquet(out_path, index=False)
    else:
        out.to_csv(out_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"[match] wrote {len(out):,} rows -> {out_path}")

    if args.objects_out:
        p = Path(args.objects_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        if _is_parquet(p):
            obj.to_parquet(p, index=False)
        else:
            obj.to_csv(p, index=False, date_format="%Y-%m-%d %H:%M:%S")
        print(f"[match] objects with motion -> {p}")

    if args.tracks_out:
        p = Path(args.tracks_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        if _is_parquet(p):
            tr.to_parquet(p, index=False)
        else:
            tr.to_csv(p, index=False, date_format="%Y-%m-%d %H:%M:%S")
        print(f"[match] tracks with motion -> {p}")

    if "obj_motion_bearing_deg" in obj.columns and "obj_flow_bearing_deg" in obj.columns:
        a = pd.to_numeric(obj["obj_motion_bearing_deg"], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(obj["obj_flow_bearing_deg"], errors="coerce").to_numpy(dtype=float)
        diff = ((a - b + 180.0) % 360.0) - 180.0
        diff = diff[np.isfinite(diff)]
        if diff.size:
            print(f"[match] motion vs flow bearing | median={float(np.nanmedian(np.abs(diff))):.1f}° "
                  f"p90={float(np.nanquantile(np.abs(diff), 0.9)):.1f}°")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
