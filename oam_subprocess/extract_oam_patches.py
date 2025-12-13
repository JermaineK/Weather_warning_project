#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_oam_patches.py

Path A: pull compact patches around candidate grid cells so an external OAM /
spiral detector can run on manageable subsets (instead of the full grid).

Design goals:
  - Input is an existing grid table (CSV/Parquet) with time/lat/lon and ilat/ilon.
  - Candidates can come from a seeds file or inline filters (prob/flag threshold).
  - Patches are saved as compressed .npz files (one per patch) plus a metadata table
    mapping patch_id -> time/lat/lon -> patch file.
  - Only reads the required columns; streaming/chunked for large inputs.

Nothing about the underlying maths is changed; this script only slices and exports
data for downstream OAM analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pq = None


def _is_parquet(path: str) -> bool:
    return Path(path).suffix.lower() in {".parquet", ".parq", ".pq"}


def _read_iter(path: str, columns: Sequence[str] | None, chunk_rows: int | None) -> Iterable[pd.DataFrame]:
    if _is_parquet(path):
        if chunk_rows and chunk_rows > 0 and pq is not None:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=int(chunk_rows), columns=list(columns) if columns else None):
                yield batch.to_pandas()
            return
        yield pd.read_parquet(path, columns=list(columns) if columns else None)
        return
    if chunk_rows and chunk_rows > 0:
        for ch in pd.read_csv(path, usecols=list(columns) if columns else None, chunksize=int(chunk_rows), low_memory=False):
            yield ch
        return
    yield pd.read_csv(path, usecols=list(columns) if columns else None, low_memory=False)


def _to_utc_naive(series: pd.Series, fmt: Optional[str]) -> pd.Series:
    raw = series.astype(str).str.strip().str.replace("Z", "", regex=False)
    if fmt:
        t = pd.to_datetime(raw, format=fmt, utc=True, errors="coerce")
    else:
        t = pd.to_datetime(raw, utc=True, errors="coerce")
    return t.dt.tz_convert(None)


def _norm_lon(series: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if mode == "none":
        return x
    if mode == "0..360":
        return (x % 360 + 360) % 360
    return ((x + 180) % 360) - 180


def _load_seeds(path: str, time_fmt: Optional[str], lon_mode: str, round_dp: int) -> Dict[Tuple[pd.Timestamp, float, float], bool]:
    seeds: Dict[Tuple[pd.Timestamp, float, float], bool] = {}
    for chunk in _read_iter(path, columns=None, chunk_rows=None):
        if "time" not in chunk.columns:
            for c in ["time_h", "datetime", "valid_time"]:
                if c in chunk.columns:
                    chunk = chunk.rename(columns={c: "time"})
                    break
        chunk["time"] = _to_utc_naive(chunk["time"], time_fmt)
        if "lat" not in chunk.columns or "lon" not in chunk.columns:
            continue
        chunk["lat_r"] = pd.to_numeric(chunk["lat"], errors="coerce").round(round_dp)
        chunk["lon_r"] = _norm_lon(pd.to_numeric(chunk["lon"], errors="coerce"), lon_mode).round(round_dp)
        for t, la, lo in chunk[["time", "lat_r", "lon_r"]].itertuples(index=False):
            if pd.isna(t) or pd.isna(la) or pd.isna(lo):
                continue
            seeds[(t, float(la), float(lo))] = True
    return seeds


def _pivot_patch(df: pd.DataFrame, vars_keep: List[str]) -> Dict[str, np.ndarray]:
    """
    Build dense arrays for the requested variables on the minimal bounding box.
    """
    lat_keys = np.sort(df["ilat"].unique())
    lon_keys = np.sort(df["ilon"].unique())
    lat_idx = {v: i for i, v in enumerate(lat_keys)}
    lon_idx = {v: i for i, v in enumerate(lon_keys)}
    shape = (len(lat_keys), len(lon_keys))

    out: Dict[str, np.ndarray] = {}
    for var in vars_keep:
        arr = np.full(shape, np.nan, dtype=float)
        vals = pd.to_numeric(df[var], errors="coerce").to_numpy()
        for (ilat, ilon), val in zip(df[["ilat", "ilon"]].itertuples(index=False, name=None), vals):
            i = lat_idx.get(ilat)
            j = lon_idx.get(ilon)
            if i is None or j is None:
                continue
            arr[i, j] = val
        out[var] = arr
    # add coordinate grids for convenience
    out["ilat_grid"] = np.repeat(lat_keys[:, None], len(lon_keys), axis=1)
    out["ilon_grid"] = np.repeat(lon_keys[None, :], len(lat_keys), axis=0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract small patches around candidate cells for OAM/spiral analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--panel", required=True, help="Grid table (CSV/Parquet) containing time/lat/lon/ilat/ilon and variables.")
    ap.add_argument("--seeds", default=None, help="Optional seeds table to define candidates (CSV/Parquet).")
    ap.add_argument("--prob-col", default=None, help="Probability column to threshold for candidates (panel).")
    ap.add_argument("--prob-thr", type=float, default=None, help="Threshold on prob-col; rows meeting it become candidates.")
    ap.add_argument("--flag-col", default=None, help="Optional 0/1 flag column; >0 marks candidates.")
    ap.add_argument("--vars", default="zeta,div,S,pdrop_nd,t2m_anom_local,gka_knee_ratio,gka_score", help="Comma-separated variables to extract.")
    ap.add_argument("--patch-radius-cells", type=int, default=2, help="+/- cells in ilat/ilon to include in the patch.")
    ap.add_argument("--grid-step-deg", type=float, default=0.25, help="Grid step (deg) used when ilat/ilon missing; used with rounding.")
    ap.add_argument("--time-col", default="time")
    ap.add_argument("--lat-col", default="lat")
    ap.add_argument("--lon-col", default="lon")
    ap.add_argument("--ilat-col", default="ilat")
    ap.add_argument("--ilon-col", default="ilon")
    ap.add_argument("--time-format", default=None, help="Optional strptime format for time parsing.")
    ap.add_argument("--normalize-lon", choices=["none", "-180..180", "0..360"], default="-180..180")
    ap.add_argument("--round-dp", type=int, default=3, help="Decimal places for lat/lon rounding when matching seeds.")
    ap.add_argument("--max-patches", type=int, default=None, help="Optional cap on number of patches extracted.")
    ap.add_argument("--chunk-rows", type=int, default=0, help="Chunk rows for streaming.")
    ap.add_argument("--out-meta", default="results/oam_patches/oam_patches_meta.parquet", help="Metadata table (Parquet/CSV).")
    ap.add_argument("--out-dir", default="results/oam_patches/patches", help="Directory to write patch .npz files.")
    ap.add_argument("--save-format", choices=["npz", "npy"], default="npz", help="Patch file format.")
    args = ap.parse_args()

    vars_keep = [v.strip() for v in args.vars.split(",") if v.strip()]
    if not vars_keep:
        raise SystemExit("No variables requested; provide --vars.")

    seeds_index: Dict[Tuple[pd.Timestamp, float, float], bool] = {}
    if args.seeds:
        print(f"[oam-patches] loading seeds from {args.seeds}")
        seeds_index = _load_seeds(args.seeds, args.time_format, args.normalize_lon, args.round_dp)
        print(f"[oam-patches] seeds loaded: {len(seeds_index):,} keys")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_rows: List[Dict] = []
    patch_id = 0
    chunk_rows = args.chunk_rows if args.chunk_rows and args.chunk_rows > 0 else None

    needed_cols = {args.time_col, args.lat_col, args.lon_col, args.ilat_col, args.ilon_col, *vars_keep}
    if args.prob_col:
        needed_cols.add(args.prob_col)
    if args.flag_col:
        needed_cols.add(args.flag_col)

    for chunk in _read_iter(args.panel, columns=list(needed_cols), chunk_rows=chunk_rows):
        if chunk.empty:
            continue
        chunk = chunk.copy()
        chunk[args.time_col] = _to_utc_naive(chunk[args.time_col], args.time_format)
        chunk[args.lat_col] = pd.to_numeric(chunk[args.lat_col], errors="coerce")
        chunk[args.lon_col] = _norm_lon(pd.to_numeric(chunk[args.lon_col], errors="coerce"), args.normalize_lon)

        # derive ilat/ilon if missing
        if args.ilat_col not in chunk:
            chunk[args.ilat_col] = (chunk[args.lat_col] / args.grid_step_deg).round().astype("int64")
        if args.ilon_col not in chunk:
            chunk[args.ilon_col] = (chunk[args.lon_col] / args.grid_step_deg).round().astype("int64")

        chunk = chunk.dropna(subset=[args.time_col, args.lat_col, args.lon_col])
        if chunk.empty:
            continue
        chunk.rename(columns={args.time_col: "time", args.lat_col: "lat", args.lon_col: "lon",
                              args.ilat_col: "ilat", args.ilon_col: "ilon"}, inplace=True)
        chunk["lat_r"] = chunk["lat"].round(args.round_dp)
        chunk["lon_r"] = chunk["lon"].round(args.round_dp)

        # candidate mask
        cand_mask = pd.Series(False, index=chunk.index)
        if seeds_index:
            keys = list(zip(chunk["time"], chunk["lat_r"], chunk["lon_r"]))
            cand_mask |= pd.Series([seeds_index.get((t, float(la), float(lo)), False) for t, la, lo in keys], index=chunk.index)
        if args.prob_col and args.prob_thr is not None and args.prob_col in chunk:
            cand_mask |= pd.to_numeric(chunk[args.prob_col], errors="coerce") >= float(args.prob_thr)
        if args.flag_col and args.flag_col in chunk:
            cand_mask |= pd.to_numeric(chunk[args.flag_col], errors="coerce").fillna(0) > 0

        candidates = chunk.loc[cand_mask].copy()
        if candidates.empty:
            continue

        # process per time slice to keep patches local
        for t, df_t in chunk.groupby("time"):
            cand_t = candidates.loc[candidates["time"] == t]
            if cand_t.empty:
                continue
            for _, row in cand_t.iterrows():
                if args.max_patches and patch_id >= args.max_patches:
                    break
                ilat0, ilon0 = int(row["ilat"]), int(row["ilon"])
                window = df_t.loc[
                    (df_t["ilat"].between(ilat0 - args.patch_radius_cells, ilat0 + args.patch_radius_cells))
                    & (df_t["ilon"].between(ilon0 - args.patch_radius_cells, ilon0 + args.patch_radius_cells))
                ]
                if window.empty:
                    continue
                patch = _pivot_patch(window[["ilat", "ilon", *vars_keep]], vars_keep)
                patch_file = out_dir / f"patch_{patch_id:07d}.{args.save_format}"
                if args.save_format == "npz":
                    np.savez_compressed(patch_file, **patch)
                else:
                    np.save(patch_file, patch)
                meta_rows.append(
                    {
                        "patch_id": patch_id,
                        "time": t,
                        "lat": float(row["lat"]),
                        "lon": float(row["lon"]),
                        "ilat": ilat0,
                        "ilon": ilon0,
                        "patch_file": str(patch_file),
                        "n_cells": len(window),
                        "vars": ",".join(vars_keep),
                    }
                )
                patch_id += 1
            if args.max_patches and patch_id >= args.max_patches:
                break
        if args.max_patches and patch_id >= args.max_patches:
            break

    if not meta_rows:
        print("[oam-patches] no patches written.")
        return

    meta = pd.DataFrame(meta_rows)
    out_meta = Path(args.out_meta)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    if out_meta.suffix.lower() in {".parquet", ".parq", ".pq"}:
        meta.to_parquet(out_meta, index=False)
    else:
        meta.to_csv(out_meta, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"[oam-patches] wrote {len(meta):,} patches -> {out_meta}")


if __name__ == "__main__":
    main()
