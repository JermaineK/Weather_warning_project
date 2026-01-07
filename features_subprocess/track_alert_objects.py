#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_alert_objects.py

Agent: convert per-cell alert probabilities into connected objects, track them
hour-to-hour, and emit lightweight propagation features. Designed to run
streaming on time-sorted tables without loading everything into memory.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover - optional
    pa = None
    pq = None


# ---------------------------------------------------------------------------#
# IO helpers                                                                 #
# ---------------------------------------------------------------------------#

def _is_parquet(path: str | Path) -> bool:
    low = str(path).lower()
    return low.endswith((".parquet", ".parq", ".pq"))


def _peek_columns(path: str | Path) -> List[str]:
    p = Path(path)
    if _is_parquet(p):
        if pq is None:
            return list(pd.read_parquet(p, nrows=1).columns)
        return list(pq.ParquetFile(p).schema.names)
    return list(pd.read_csv(p, nrows=1, low_memory=False).columns)


def _iter_batches(path: str | Path, columns: Sequence[str], chunk_rows: int, parquet_rows: int) -> Iterable[pd.DataFrame]:
    if _is_parquet(path):
        if pq is None or pa is None:
            yield pd.read_parquet(path, columns=list(columns))
            return
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=parquet_rows or chunk_rows or None, columns=list(columns)):
            yield batch.to_pandas()
    else:
        kwargs = {"usecols": list(columns), "low_memory": False, "parse_dates": ["time"] if "time" in columns else None}
        for chunk in pd.read_csv(path, chunksize=chunk_rows or None, **{k: v for k, v in kwargs.items() if v is not None}):
            yield chunk


def _write_stream(path: str | Path, df: pd.DataFrame, writer, first: bool):
    p = Path(path)
    if p is None:
        return writer, first
    p.parent.mkdir(parents=True, exist_ok=True)
    if _is_parquet(p):
        if pq is None or pa is None:
            if not first and p.exists():
                raise SystemExit("pyarrow required for streaming parquet writes.")
            df.to_parquet(p, index=False)
            return writer, False
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(p, tbl.schema)
        writer.write_table(tbl)
        return writer, False
    comp = "gzip" if str(p).lower().endswith(".gz") else "infer"
    mode = "w" if first else "a"
    header = first
    df.to_csv(p, index=False, mode=mode, header=header, compression=comp, date_format="%Y-%m-%d %H:%M:%S")
    return writer, False


# ---------------------------------------------------------------------------#
# Component + tracking helpers                                               #
# ---------------------------------------------------------------------------#

def _neighbor_offsets(connectivity: int) -> List[Tuple[int, int]]:
    base = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if connectivity == 4:
        return base
    diag = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    return base + diag


def _components(indices: List[Tuple[int, int]], connectivity: int) -> List[List[Tuple[int, int]]]:
    mapping = {coord: idx for idx, coord in enumerate(indices)}
    seen = set()
    comps: List[List[Tuple[int, int]]] = []
    nbrs = _neighbor_offsets(connectivity)
    for coord in indices:
        if coord in seen:
            continue
        stack = [coord]
        comp: List[Tuple[int, int]] = []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            ci, cj = cur
            for di, dj in nbrs:
                nb = (ci + di, cj + dj)
                if nb in mapping and nb not in seen:
                    stack.append(nb)
        comps.append(comp)
    return comps


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    # low-cost haversine for small deltas
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2 * r * math.asin(math.sqrt(max(a, 0.0)))


def _match_component(
    comp_cells: set,
    centroid: Tuple[float, float],
    prev_components: List[Dict],
    time_gap_hours: float,
    centroid_max_km: float,
    current_time: pd.Timestamp,
) -> Optional[Dict]:
    best: Optional[Dict] = None
    best_overlap = 0
    for obj in prev_components:
        gap_h = (current_time - obj["time"]).total_seconds() / 3600.0
        if gap_h > time_gap_hours:
            continue
        overlap = len(comp_cells & obj["cells"])
        if overlap > best_overlap and overlap > 0:
            best_overlap = overlap
            best = obj
    if best:
        return best

    # fallback: nearest centroid if within max distance
    nearest: Optional[Dict] = None
    nearest_dist = float("inf")
    lat, lon = centroid
    for obj in prev_components:
        gap_h = (current_time - obj["time"]).total_seconds() / 3600.0
        if gap_h > time_gap_hours:
            continue
        dist = _haversine_km(lat, lon, obj["centroid"][0], obj["centroid"][1])
        if dist < nearest_dist and dist <= centroid_max_km:
            nearest_dist = dist
            nearest = obj
    return nearest


def _prepare_keys(df: pd.DataFrame, ilat_col: str, ilon_col: str, lat_col: str, lon_col: str) -> Tuple[np.ndarray, np.ndarray]:
    if ilat_col in df.columns and ilon_col in df.columns:
        return (
            pd.to_numeric(df[ilat_col], errors="coerce").astype("Int64").to_numpy(),
            pd.to_numeric(df[ilon_col], errors="coerce").astype("Int64").to_numpy(),
        )
    # fallback: dense ranks per time slice
    lat_codes, _ = pd.factorize(df[lat_col], sort=True)
    lon_codes, _ = pd.factorize(df[lon_col], sort=True)
    return lat_codes.astype(np.int64), lon_codes.astype(np.int64)


def _select_mask_column(path: str | Path, args, chunk_rows: int, parquet_rows: int) -> Tuple[str, float]:
    candidates = [args.mask_col]
    if args.fallback_mask_cols:
        extras = [c.strip() for c in str(args.fallback_mask_cols).split(",") if c.strip()]
        candidates.extend(extras)
    candidates = [c for i, c in enumerate(candidates) if c and c not in candidates[:i]]
    cols_available = set(_peek_columns(path))
    candidates = [c for c in candidates if c in cols_available]
    if not candidates:
        raise SystemExit(
            f"[track] no usable mask columns found. Available columns: {sorted(cols_available)[:20]}"
        )

    sample_cols = list({args.time_col, args.lat_col, args.lon_col, args.ilat_col, args.ilon_col, *candidates})
    stats: Dict[str, Dict[str, float]] = {}
    values_cache: Dict[str, np.ndarray] = {}
    for chunk in _iter_batches(path, sample_cols, chunk_rows or 200_000, parquet_rows or 200_000):
        if chunk.empty:
            continue
        for col in candidates:
            if col not in chunk.columns:
                continue
            vals = pd.to_numeric(chunk[col], errors="coerce").to_numpy()
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                continue
            stats[col] = {
                "pos": float((finite >= float(args.threshold)).sum()),
                "max": float(np.nanmax(finite)),
                "mean": float(np.nanmean(finite)),
            }
            values_cache[col] = finite
        break

    if not stats:
        return args.mask_col, float(args.threshold)

    best = max(stats.items(), key=lambda kv: kv[1]["pos"])
    best_col, best_stats = best
    if best_stats["pos"] > 0:
        return best_col, float(args.threshold)

    # No positives at the provided threshold; fall back to a quantile threshold if possible.
    vals = values_cache.get(best_col, np.array([], dtype=float))
    if vals.size == 0 or not np.isfinite(best_stats["max"]):
        return best_col, float(args.threshold)
    q = float(getattr(args, "fallback_quantile", 0.99))
    q = min(max(q, 0.5), 0.999)
    thr = float(np.nanquantile(vals, q))
    if not np.isfinite(thr):
        thr = float(best_stats["max"])
    print(
        f"[track] mask '{best_col}' has no positives at thr={args.threshold}; "
        f"using fallback threshold={thr:.4f} (q={q})."
    )
    return best_col, thr


# ---------------------------------------------------------------------------#
# Streaming per-time processor                                               #
# ---------------------------------------------------------------------------#

def _iter_time_groups(path: str, columns: Sequence[str], chunk_rows: int, parquet_rows: int, time_col: str):
    buffer = pd.DataFrame()
    for chunk in _iter_batches(path, columns, chunk_rows, parquet_rows):
        if chunk.empty:
            continue
        chunk[time_col] = pd.to_datetime(chunk[time_col])
        if not buffer.empty:
            chunk = pd.concat([buffer, chunk], axis=0, ignore_index=True)
            buffer = pd.DataFrame()
        chunk.sort_values(time_col, inplace=True)
        times = list(chunk[time_col].unique())
        if not times:
            continue
        for t in times[:-1]:
            yield t, chunk[chunk[time_col] == t].copy()
        buffer = chunk[chunk[time_col] == times[-1]].copy()
    if not buffer.empty:
        buffer.sort_values(time_col, inplace=True)
        for t, g in buffer.groupby(time_col):
            yield t, g.copy()


def track_objects(path: str, args) -> None:
    chunk_rows = args.chunk_rows or getattr(args, "chunksize", 0) or 0
    parquet_rows = args.parquet_rows or chunk_rows
    mask_col, threshold = _select_mask_column(path, args, chunk_rows, parquet_rows)
    if mask_col != args.mask_col:
        print(f"[track] using mask column '{mask_col}' (requested '{args.mask_col}')")
    args.mask_col = mask_col
    args.threshold = float(threshold)
    cols = {
        args.time_col,
        args.lat_col,
        args.lon_col,
        args.ilat_col,
        args.ilon_col,
        args.mask_col,
    }
    batches = _iter_time_groups(path, list(cols), chunk_rows, parquet_rows, args.time_col)

    prev_components: List[Dict] = []
    next_obj_id = 1
    object_rows: List[Dict] = []
    cell_writer = None
    cell_first = True

    for t_val, df_t in batches:
        scores = pd.to_numeric(df_t[args.mask_col], errors="coerce").fillna(0.0)
        keep_mask = scores >= float(args.threshold)
        if not keep_mask.any():
            prev_components = []
            continue

        df_sel = df_t.loc[keep_mask].copy()
        lat_idx, lon_idx = _prepare_keys(df_sel, args.ilat_col, args.ilon_col, args.lat_col, args.lon_col)
        coords = list(zip(lat_idx.tolist(), lon_idx.tolist()))
        comps = _components(coords, args.connectivity)

        # map back to row indices
        coord_to_pos = {c: i for i, c in enumerate(coords)}

        new_prev: List[Dict] = []
        for comp in comps:
            rows_idx = [coord_to_pos[c] for c in comp]
            comp_cells = set(comp)
            lat_vals = df_sel.iloc[rows_idx][args.lat_col].to_numpy()
            lon_vals = df_sel.iloc[rows_idx][args.lon_col].to_numpy()
            prob_vals = pd.to_numeric(df_sel[args.mask_col], errors="coerce").to_numpy()
            # intensity mean on component
            intensity = float(np.nanmean(prob_vals[rows_idx]))
            centroid_lat = float(np.nanmean(lat_vals))
            centroid_lon = float(np.nanmean(lon_vals))

            matched = _match_component(
                comp_cells,
                (centroid_lat, centroid_lon),
                prev_components,
                args.max_gap_hours,
                args.max_centroid_km,
                t_val,
            )
            if matched:
                obj_id = matched["id"]
                area_prev = matched["area"]
                dt_hours = max((t_val - matched["time"]).total_seconds() / 3600.0, 1e-6)
                dlat = centroid_lat - matched["centroid"][0]
                dlon = centroid_lon - matched["centroid"][1]
                speed_deg = math.sqrt(dlat * dlat + dlon * dlon) / dt_hours
                heading_deg = math.degrees(math.atan2(dlon, dlat))
            else:
                obj_id = next_obj_id
                next_obj_id += 1
                area_prev = 0
                speed_deg = 0.0
                heading_deg = 0.0

            area = len(rows_idx)
            growth = area - area_prev
            object_rows.append(
                {
                    "object_id": obj_id,
                    args.time_col: t_val,
                    "obj_area_cells": area,
                    "obj_centroid_lat": centroid_lat,
                    "obj_centroid_lon": centroid_lon,
                    "obj_intensity_mean": intensity,
                    "obj_growth_rate": growth,
                    "obj_speed_deg_per_h": speed_deg,
                    "obj_heading_deg": heading_deg,
                }
            )

            if args.join_out:
                comp_df = df_sel.iloc[rows_idx].copy()
                comp_df["object_id"] = obj_id
                comp_df["obj_area_cells"] = area
                comp_df["obj_centroid_lat"] = centroid_lat
                comp_df["obj_centroid_lon"] = centroid_lon
                comp_df["obj_intensity_mean"] = intensity
                comp_df["distance_to_centroid_deg"] = np.sqrt(
                    (pd.to_numeric(comp_df[args.lat_col], errors="coerce") - centroid_lat) ** 2
                    + (pd.to_numeric(comp_df[args.lon_col], errors="coerce") - centroid_lon) ** 2
                )
                cell_writer, cell_first = _write_stream(args.join_out, comp_df, cell_writer, cell_first)

            new_prev.append(
                {
                    "id": obj_id,
                    "time": t_val,
                    "cells": comp_cells,
                    "centroid": (centroid_lat, centroid_lon),
                    "area": area,
                }
            )
        prev_components = new_prev

    if cell_writer is not None and hasattr(cell_writer, "close"):
        cell_writer.close()
    # write object table
    if args.objects_out:
        obj_df = pd.DataFrame(object_rows)
        out_path = Path(args.objects_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if _is_parquet(out_path):
            obj_df.to_parquet(out_path, index=False)
        else:
            obj_df.to_csv(out_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
        print(f"[track] wrote {len(obj_df):,} object rows -> {out_path}")
    else:
        print(f"[track] skipped object table (objects_out not set); objects computed: {len(object_rows):,}")


# ---------------------------------------------------------------------------#
# CLI                                                                        #
# ---------------------------------------------------------------------------#

def parse_args():
    ap = argparse.ArgumentParser(
        description="Track alert objects from per-cell probabilities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--infile", required=True, help="Input table with per-cell scores.")
    ap.add_argument("--mask-col", default="P_final", help="Column to threshold for object mask.")
    ap.add_argument("--threshold", type=float, default=0.6, help="Threshold for mask_col.")
    ap.add_argument(
        "--fallback-mask-cols",
        default="alert_mask,alert_final,alert_base,alert_throttled,alert,P_final,P_base,prob_viable,prob",
        help="Fallback mask columns (comma-separated) if mask_col is missing or empty.",
    )
    ap.add_argument(
        "--fallback-quantile",
        type=float,
        default=0.99,
        help="Quantile for fallback threshold when no positives are found.",
    )
    ap.add_argument("--objects-out", default="results/objects.parquet", help="Object-level output table.")
    ap.add_argument("--join-out", default=None, help="Optional per-cell table with object_id + object stats.")
    ap.add_argument("--connectivity", type=int, default=8, choices=[4, 8], help="Grid connectivity for components.")
    ap.add_argument("--max-gap-hours", type=float, default=3.0, help="Max time gap (hours) to link objects.")
    ap.add_argument("--max-centroid-km", type=float, default=150.0, help="Max centroid distance (km) for linking.")
    ap.add_argument("--time-col", default="time")
    ap.add_argument("--lat-col", default="lat")
    ap.add_argument("--lon-col", default="lon")
    ap.add_argument("--ilat-col", default="ilat")
    ap.add_argument("--ilon-col", default="ilon")
    ap.add_argument("--chunk-rows", type=int, default=200_000, help="Chunk size for CSV streaming.")
    ap.add_argument("--chunksize", type=int, default=200_000, help="Alias for --chunk-rows.")
    ap.add_argument("--parquet-rows", type=int, default=200_000, help="Batch size for parquet streaming.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they already exist.")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.objects_out:
        obj_path = Path(args.objects_out)
        if obj_path.exists() and not args.overwrite:
            print(f"[track] objects_out exists; skipping (use --overwrite): {obj_path}")
            return
    if args.join_out:
        join_path = Path(args.join_out)
        if join_path.exists() and not args.overwrite:
            print(f"[track] join_out exists; skipping (use --overwrite): {join_path}")
            return
    track_objects(args.infile, args)


if __name__ == "__main__":
    main()
