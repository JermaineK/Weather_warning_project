#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
objects_by_hour.py

Agent: build object-level components for reporting without changing physics math.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # optional, for efficient parquet streaming
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover - optional
    pa = None
    pq = None


# --------------------------- IO helpers ---------------------------

def _is_parquet(path: str | Path) -> bool:
    low = str(path).lower()
    return low.endswith((".parquet", ".parq", ".pq", ".pqt"))


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


def _peek_columns(path: str | Path) -> List[str]:
    p = Path(path)
    if _is_parquet(p):
        if pq is None:
            return list(pd.read_parquet(p, columns=None).columns)
        return list(pq.ParquetFile(p).schema.names)
    return list(pd.read_csv(p, nrows=0).columns)


# --------------------------- component helpers ---------------------------

def _neighbor_offsets(connectivity: int) -> List[Tuple[int, int]]:
    base = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if connectivity == 4:
        return base
    diag = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    return base + diag


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2 * r * math.asin(math.sqrt(max(a, 0.0)))


def _majority_filter(coords: List[Tuple[int, int]], connectivity: int, k: int) -> List[Tuple[int, int]]:
    if not coords or k <= 1:
        return coords
    active = set(coords)
    nbrs = _neighbor_offsets(connectivity)
    keep: List[Tuple[int, int]] = []
    for ci, cj in coords:
        count = 1
        for di, dj in nbrs:
            if (ci + di, cj + dj) in active:
                count += 1
        if count >= k:
            keep.append((ci, cj))
    return keep


def _assign_tracks(
    active_tracks: Dict[int, Dict[str, float]],
    current: List[Dict[str, object]],
    max_km: float,
) -> Dict[int, int]:
    if not active_tracks or not current:
        return {}
    candidates: List[Tuple[float, int, int]] = []
    for track_id, info in active_tracks.items():
        lat0 = float(info.get("lat", np.nan))
        lon0 = float(info.get("lon", np.nan))
        if not np.isfinite(lat0) or not np.isfinite(lon0):
            continue
        for idx, obj in enumerate(current):
            lat1 = float(obj.get("centroid_lat", np.nan))
            lon1 = float(obj.get("centroid_lon", np.nan))
            if not np.isfinite(lat1) or not np.isfinite(lon1):
                continue
            dist = _haversine_km(lat0, lon0, lat1, lon1)
            if dist <= max_km:
                candidates.append((dist, track_id, idx))
    candidates.sort(key=lambda x: x[0])
    assigned_tracks: set[int] = set()
    assigned_objs: set[int] = set()
    out: Dict[int, int] = {}
    for _, track_id, idx in candidates:
        if track_id in assigned_tracks or idx in assigned_objs:
            continue
        out[idx] = track_id
        assigned_tracks.add(track_id)
        assigned_objs.add(idx)
    return out


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


def _prepare_keys(df: pd.DataFrame, ilat_col: str, ilon_col: str, lat_col: str, lon_col: str) -> Tuple[np.ndarray, np.ndarray]:
    if ilat_col in df.columns and ilon_col in df.columns:
        return (
            pd.to_numeric(df[ilat_col], errors="coerce").to_numpy(dtype=np.int64),
            pd.to_numeric(df[ilon_col], errors="coerce").to_numpy(dtype=np.int64),
        )
    lat_codes, _ = pd.factorize(df[lat_col], sort=True)
    lon_codes, _ = pd.factorize(df[lon_col], sort=True)
    return lat_codes.astype(np.int64), lon_codes.astype(np.int64)


def _principal_axis(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if x.size < 2:
        return float("nan"), float("nan")
    coords = np.column_stack([x, y]).astype(float)
    coords = coords - coords.mean(axis=0)
    if not np.isfinite(coords).any():
        return float("nan"), float("nan")
    cov = np.cov(coords, rowvar=False)
    try:
        vals, vecs = np.linalg.eigh(cov)
    except Exception:
        return float("nan"), float("nan")
    order = np.argsort(vals)[::-1]
    if vals[order[1]] <= 0 or not np.isfinite(vals[order[0]]):
        elong = float("inf")
    else:
        elong = float(vals[order[0]] / vals[order[1]])
    vec = vecs[:, order[0]]
    bearing = (math.degrees(math.atan2(vec[0], vec[1])) + 360.0) % 360.0
    return float(bearing), float(elong)


def _topk_mean(x: np.ndarray, k: int) -> float:
    if x.size == 0 or k <= 0:
        return float("nan")
    vals = x[np.isfinite(x)]
    if vals.size == 0:
        return float("nan")
    k = min(k, vals.size)
    part = np.partition(vals, -k)[-k:]
    return float(np.nanmean(part))


# --------------------------- streaming group by time ---------------------------

def _iter_time_groups(path: str, columns: Sequence[str], chunk_rows: int, parquet_rows: int, time_col: str):
    buffer = pd.DataFrame()
    for chunk in _iter_batches(path, columns, chunk_rows, parquet_rows):
        if chunk.empty:
            continue
        times = pd.to_datetime(chunk[time_col], utc=True, errors="coerce")
        bad = int(times.isna().sum())
        if bad:
            raise SystemExit(f"[objects] invalid {time_col} values: {bad} rows failed time parsing.")
        chunk[time_col] = times.dt.tz_localize(None)
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


# --------------------------- main logic ---------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract connected object components per hour for reporting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--infile", required=True, help="Input table with per-cell alerts/probabilities.")
    ap.add_argument("--objects-out", default="results/objects/objects_by_hour.parquet", help="Object-level output table.")
    ap.add_argument("--cells-out", default=None, help="Optional per-cell table with object_id.")
    ap.add_argument("--mask-col", default="alert_final", help="Binary flag column for candidate cells.")
    ap.add_argument("--score-col", default="prob_viable", help="Score column for ranking/probability stats.")
    ap.add_argument("--threshold", type=float, default=None, help="Score threshold (used with score-col).")
    ap.add_argument("--min-neighbors", type=int, default=1, help="Minimum cells per component (1 keeps all).")
    ap.add_argument("--min-area-cells", type=int, default=1, help="Minimum object area (cells) before filtering.")
    ap.add_argument("--persist-hours-small", type=int, default=0, help="Keep small objects only if they persist this many hours.")
    ap.add_argument("--persist-link-km", type=float, default=75.0, help="Link radius for persistence tracking (km).")
    ap.add_argument("--morphology", choices=["none", "majority"], default="none", help="Optional mask smoothing.")
    ap.add_argument("--morph-k", type=int, default=3, help="Neighbor threshold for majority smoothing.")
    ap.add_argument("--connectivity", type=int, choices=[4, 8], default=8, help="Grid connectivity.")
    ap.add_argument("--core-k", type=int, default=10, help="Top-k cells used for core score.")
    ap.add_argument("--rejects-out", default=None, help="Optional per-hour rejected objects summary output.")
    ap.add_argument("--extra-cols", default=None, help="Comma-separated extra columns to aggregate.")
    ap.add_argument("--time-col", default="time")
    ap.add_argument("--lat-col", default="lat")
    ap.add_argument("--lon-col", default="lon")
    ap.add_argument("--ilat-col", default="ilat")
    ap.add_argument("--ilon-col", default="ilon")
    ap.add_argument("--chunk-rows", dest="chunk_rows", type=int, default=200_000, help="Chunk size for CSV streaming.")
    ap.add_argument("--chunksize", dest="chunk_rows", type=int, default=200_000, help="Alias for --chunk-rows.")
    ap.add_argument("--parquet-rows", dest="parquet_rows", type=int, default=200_000, help="Batch size for parquet streaming.")
    args = ap.parse_args()

    min_area_cells = max(1, int(args.min_neighbors), int(args.min_area_cells))
    persist_hours_small = max(0, int(args.persist_hours_small))
    use_persist = persist_hours_small > 0

    cols = _peek_columns(args.infile)
    base_cols = {args.time_col, args.lat_col, args.lon_col}
    missing_base = [c for c in base_cols if c not in cols]
    if missing_base:
        raise SystemExit(f"[objects] missing required columns: {missing_base}")

    has_mask = args.mask_col in cols
    has_score = args.score_col in cols
    if not has_mask and not has_score:
        raise SystemExit(
            f"[objects] missing both mask-col '{args.mask_col}' and score-col '{args.score_col}'. "
            "Provide at least one to avoid treating all cells as candidates."
        )
    if not has_mask and args.threshold is None:
        raise SystemExit(
            "[objects] no mask-col available and --threshold not set; "
            "refusing to treat all cells as candidates."
        )
    if args.threshold is not None and not has_score:
        raise SystemExit(f"[objects] threshold set but score-col '{args.score_col}' not found.")

    # Aggregate columns to compute per-object mean/max
    candidate_cols = [c for c in cols if c.startswith("prob_")]
    extra_named = [c for c in (args.extra_cols or "").split(",") if c.strip()]
    for c in ["G_struct", "S_shear", "E_energy", "gka_F", "SFI", "SFI2", "u10", "v10"]:
        if c in cols:
            extra_named.append(c)
    agg_cols = sorted(set(candidate_cols + extra_named))

    needed_cols = set(base_cols)
    needed_cols.update({args.ilat_col, args.ilon_col, args.mask_col, args.score_col})
    # Keep row_id when available so downstream flow enrichment can join to full panels.
    if "row_id" in cols:
        needed_cols.add("row_id")
    needed_cols.update(agg_cols)
    needed_cols = [c for c in needed_cols if c in cols]

    obj_writer = None
    obj_first = True
    obj_written = 0
    cell_writer = None
    cell_first = True

    batches = _iter_time_groups(args.infile, needed_cols, args.chunk_rows, args.parquet_rows, args.time_col)
    next_obj_id = 1
    next_track_id = 1
    active_tracks: Dict[int, Dict[str, object]] = {}
    reject_rows: List[Dict[str, object]] = []

    def _write_objects(rows: List[Dict[str, object]]) -> None:
        nonlocal obj_writer, obj_first, obj_written
        if not rows:
            return
        obj_df = pd.DataFrame(rows)
        obj_writer, obj_first = _write_stream(args.objects_out, obj_df, obj_writer, obj_first)
        obj_written += len(obj_df)

    for t_val, df_t in batches:
        if df_t.empty:
            continue
        scores = pd.to_numeric(df_t[args.score_col], errors="coerce") if has_score else pd.Series(np.nan, index=df_t.index)
        mask = pd.Series(False, index=df_t.index)
        if has_mask:
            mask |= pd.to_numeric(df_t[args.mask_col], errors="coerce").fillna(0).astype(int) > 0
        if args.threshold is not None and has_score:
            mask |= scores >= float(args.threshold)
        if not mask.any():
            continue

        df_sel = df_t.loc[mask].copy()
        df_sel[args.lat_col] = pd.to_numeric(df_sel[args.lat_col], errors="coerce")
        df_sel[args.lon_col] = pd.to_numeric(df_sel[args.lon_col], errors="coerce")
        df_sel = df_sel.dropna(subset=[args.lat_col, args.lon_col]).reset_index(drop=True)
        if args.ilat_col in df_sel.columns and args.ilon_col in df_sel.columns:
            df_sel[args.ilat_col] = pd.to_numeric(df_sel[args.ilat_col], errors="coerce")
            df_sel[args.ilon_col] = pd.to_numeric(df_sel[args.ilon_col], errors="coerce")
            df_sel = df_sel.dropna(subset=[args.ilat_col, args.ilon_col]).reset_index(drop=True)
        if df_sel.empty:
            continue

        lat_idx, lon_idx = _prepare_keys(df_sel, args.ilat_col, args.ilon_col, args.lat_col, args.lon_col)
        coords = list(zip(lat_idx.tolist(), lon_idx.tolist()))
        if args.morphology == "majority":
            keep_coords = _majority_filter(coords, args.connectivity, int(args.morph_k))
            if not keep_coords:
                continue
            keep_set = set(keep_coords)
            keep_idx = [i for i, c in enumerate(coords) if c in keep_set]
            df_sel = df_sel.iloc[keep_idx].reset_index(drop=True)
            lat_idx = lat_idx[keep_idx]
            lon_idx = lon_idx[keep_idx]
            coords = [coords[i] for i in keep_idx]

        comps = _components(coords, args.connectivity)
        coord_to_pos = {c: i for i, c in enumerate(coords)}

        lat_vals = df_sel[args.lat_col].to_numpy()
        lon_vals = df_sel[args.lon_col].to_numpy()
        score_vals = pd.to_numeric(df_sel[args.score_col], errors="coerce").to_numpy() if has_score else np.full(len(df_sel), np.nan)
        agg_arrays = {c: pd.to_numeric(df_sel[c], errors="coerce").to_numpy() for c in agg_cols if c in df_sel.columns}

        current_objs: List[Dict[str, object]] = []
        for comp in comps:
            rows_idx = [coord_to_pos[c] for c in comp]
            rows_idx_arr = np.asarray(rows_idx, dtype=np.int64)
            lat_c = lat_vals[rows_idx_arr]
            lon_c = lon_vals[rows_idx_arr]
            area = int(len(rows_idx_arr))

            centroid_lat = float(np.nanmean(lat_c))
            centroid_lon = float(np.nanmean(lon_c))
            lat_min = float(np.nanmin(lat_c))
            lat_max = float(np.nanmax(lat_c))
            lon_min = float(np.nanmin(lon_c))
            lon_max = float(np.nanmax(lon_c))

            score_sub = score_vals[rows_idx_arr]
            score_max = float(np.nanmax(score_sub)) if np.isfinite(score_sub).any() else float("nan")
            score_mean = float(np.nanmean(score_sub)) if np.isfinite(score_sub).any() else float("nan")
            score_topk = _topk_mean(score_sub, int(args.core_k))

            if np.isfinite(score_sub).any():
                core_idx = int(np.nanargmax(score_sub))
                core_lat = float(lat_c[core_idx])
                core_lon = float(lon_c[core_idx])
            else:
                core_lat = centroid_lat
                core_lon = centroid_lon

            if args.ilat_col in df_sel.columns and args.ilon_col in df_sel.columns:
                axis_bearing, axis_elong = _principal_axis(
                    pd.to_numeric(df_sel.iloc[rows_idx_arr][args.ilon_col], errors="coerce").to_numpy(),
                    pd.to_numeric(df_sel.iloc[rows_idx_arr][args.ilat_col], errors="coerce").to_numpy(),
                )
            else:
                axis_bearing, axis_elong = _principal_axis(lon_c, lat_c)

            row: Dict[str, object] = {
                "object_id": next_obj_id,
                args.time_col: t_val,
                "obj_area_cells": area,
                "obj_centroid_lat": centroid_lat,
                "obj_centroid_lon": centroid_lon,
                "obj_lat_min": lat_min,
                "obj_lat_max": lat_max,
                "obj_lon_min": lon_min,
                "obj_lon_max": lon_max,
                "obj_score_max": score_max,
                "obj_score_mean": score_mean,
                "obj_score_topk_mean": score_topk,
                "obj_core_lat": core_lat,
                "obj_core_lon": core_lon,
                "obj_axis_bearing_deg": axis_bearing,
                "obj_axis_elongation": axis_elong,
            }

            for col, arr in agg_arrays.items():
                sub = arr[rows_idx_arr]
                if np.isfinite(sub).any():
                    row[f"obj_{col}_mean"] = float(np.nanmean(sub))
                    row[f"obj_{col}_max"] = float(np.nanmax(sub))
                else:
                    row[f"obj_{col}_mean"] = float("nan")
                    row[f"obj_{col}_max"] = float("nan")

            comp_df = None
            if args.cells_out:
                comp_df = df_sel.iloc[rows_idx_arr].copy()
                comp_df["object_id"] = next_obj_id
                comp_df["obj_area_cells"] = area
                comp_df["obj_centroid_lat"] = centroid_lat
                comp_df["obj_centroid_lon"] = centroid_lon
                comp_df["obj_score_max"] = score_max

            current_objs.append(
                {
                    "row": row,
                    "rows_idx_arr": rows_idx_arr,
                    "centroid_lat": centroid_lat,
                    "centroid_lon": centroid_lon,
                    "area": area,
                    "comp_df": comp_df,
                }
            )
            next_obj_id += 1

        # Agent: persist filter keeps small objects only after sustained hours.
        if not use_persist:
            rejected_small = 0
            kept_total = 0
            kept_rows: List[Dict[str, object]] = []
            for obj in current_objs:
                area = int(obj.get("area", 0))
                row = obj["row"]
                if area >= min_area_cells:
                    kept_total += 1
                    kept_rows.append(row)
                    if args.cells_out and obj.get("comp_df") is not None:
                        comp_df = obj["comp_df"]
                        cell_writer, cell_first = _write_stream(args.cells_out, comp_df, cell_writer, cell_first)
                else:
                    rejected_small += 1
            _write_objects(kept_rows)
            if args.rejects_out:
                reject_rows.append(
                    {
                        "time": t_val,
                        "objects_total": len(current_objs),
                        "kept_total": kept_total,
                        "kept_small": 0,
                        "rejected_small": rejected_small,
                    }
                )
            continue

        assignments = _assign_tracks(active_tracks, current_objs, float(args.persist_link_km))
        new_active: Dict[int, Dict[str, object]] = {}
        rejected_small = 0
        kept_small = 0
        kept_total = 0
        kept_rows: List[Dict[str, object]] = []

        for idx, obj in enumerate(current_objs):
            track_id = assignments.get(idx)
            buffer: List[Dict[str, object]] = []
            if track_id is None:
                track_id = next_track_id
                next_track_id += 1
                track_len = 1
            else:
                prev = active_tracks.get(track_id, {})
                track_len = int(prev.get("len", 0)) + 1
                buffer = list(prev.get("buffer", []))

            area = int(obj.get("area", 0))
            row = obj["row"]
            row["obj_track_id"] = track_id
            row["obj_track_len_h"] = track_len

            keep = False
            if area >= min_area_cells:
                keep = True
                if buffer:
                    buffer = []
            elif use_persist and track_len >= persist_hours_small:
                keep = True

            if keep:
                kept_total += 1
                if area < min_area_cells:
                    kept_small += 1
                if buffer and track_len >= persist_hours_small:
                    kept_rows.extend(buffer)
                    buffer = []
                kept_rows.append(row)
                if args.cells_out and obj.get("comp_df") is not None:
                    comp_df = obj["comp_df"]
                    cell_writer, cell_first = _write_stream(args.cells_out, comp_df, cell_writer, cell_first)
            else:
                rejected_small += 1
                buffer.append(row)

            new_active[track_id] = {
                "lat": obj.get("centroid_lat", np.nan),
                "lon": obj.get("centroid_lon", np.nan),
                "len": track_len,
                "buffer": buffer,
            }

        active_tracks = new_active
        if args.rejects_out:
            reject_rows.append(
                {
                    "time": t_val,
                    "objects_total": len(current_objs),
                    "kept_total": kept_total,
                    "kept_small": kept_small,
                    "rejected_small": rejected_small,
                }
            )
        _write_objects(kept_rows)

    if cell_writer is not None and hasattr(cell_writer, "close"):
        cell_writer.close()
    if obj_writer is not None and hasattr(obj_writer, "close"):
        obj_writer.close()

    out_path = Path(args.objects_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if obj_written == 0:
        empty = pd.DataFrame()
        if _is_parquet(out_path):
            empty.to_parquet(out_path, index=False)
        else:
            empty.to_csv(out_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"[objects] wrote {obj_written:,} rows -> {out_path}")
    if args.cells_out:
        print(f"[objects] cells with object_id -> {args.cells_out}")
    if args.rejects_out:
        rej_path = Path(args.rejects_out)
        rej_path.parent.mkdir(parents=True, exist_ok=True)
        rej_df = pd.DataFrame(reject_rows)
        if _is_parquet(rej_path):
            rej_df.to_parquet(rej_path, index=False)
        else:
            rej_df.to_csv(rej_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
        print(f"[objects] rejects summary -> {rej_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
