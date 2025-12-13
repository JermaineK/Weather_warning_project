#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_spiral_metafield.py

Path B: build a coarse “spiral meta” field per hour by aggregating base fields
onto a coarse grid. This does NOT compute OAM itself; it prepares low-res maps
that can be joined back to the main panel or fed into external spiral detectors.

Outputs a table with one row per (time, lat_bin, lon_bin) carrying coarse stats
for requested variables (mean, std, mean_abs, count).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pa = None
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


def _norm_lon(series: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if mode == "none":
        return x
    if mode == "0..360":
        return (x % 360 + 360) % 360
    return ((x + 180) % 360) - 180


def _to_utc_naive(series: pd.Series, fmt: str | None) -> pd.Series:
    raw = series.astype(str).str.strip().str.replace("Z", "", regex=False)
    if fmt:
        t = pd.to_datetime(raw, format=fmt, utc=True, errors="coerce")
    else:
        t = pd.to_datetime(raw, utc=True, errors="coerce")
    return t.dt.tz_convert(None)


def write_out(path: Path, df: pd.DataFrame, writer_state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_parquet(str(path)):
        if writer_state.get("writer") is None:
            writer_state["writer"] = pq.ParquetWriter(path, pa.Table.from_pandas(df, preserve_index=False).schema)
        writer_state["writer"].write_table(pa.Table.from_pandas(df, preserve_index=False))
    else:
        mode = "a" if writer_state.get("written") else "w"
        header = not writer_state.get("written")
        df.to_csv(path, mode=mode, header=header, index=False, date_format="%Y-%m-%d %H:%M:%S")
        writer_state["written"] = True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build coarse spiral meta-fields per hour (coarse stats on zeta/div/etc.).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--panel", required=True, help="Input grid (CSV/Parquet) with time/lat/lon and variables.")
    ap.add_argument("--vars", default="zeta,div,S,pdrop_nd,t2m_anom_local", help="Comma-separated variables to aggregate.")
    ap.add_argument("--coarse-step-deg", type=float, default=1.0, help="Size of coarse bins in degrees.")
    ap.add_argument("--time-col", default="time")
    ap.add_argument("--lat-col", default="lat")
    ap.add_argument("--lon-col", default="lon")
    ap.add_argument("--time-format", default=None, help="Optional strptime format for time parsing.")
    ap.add_argument("--normalize-lon", choices=["none", "-180..180", "0..360"], default="-180..180")
    ap.add_argument("--chunk-rows", type=int, default=0, help="Chunk rows for streaming input.")
    ap.add_argument("--out", default="results/oam_meta/spiral_metafield.parquet", help="Output path (Parquet/CSV).")
    args = ap.parse_args()

    coarse = float(args.coarse_step_deg)
    vars_keep = [v.strip() for v in args.vars.split(",") if v.strip()]
    if not vars_keep:
        raise SystemExit("No variables requested; provide --vars.")

    needed = {args.time_col, args.lat_col, args.lon_col, *vars_keep}
    chunk_rows = args.chunk_rows if args.chunk_rows and args.chunk_rows > 0 else None

    out_path = Path(args.out)
    writer_state: dict = {}

    def process(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["time"] = _to_utc_naive(df[args.time_col], args.time_format)
        df["lat"] = pd.to_numeric(df[args.lat_col], errors="coerce")
        df["lon"] = _norm_lon(pd.to_numeric(df[args.lon_col], errors="coerce"), args.normalize_lon)
        df = df.dropna(subset=["time", "lat", "lon"])
        if df.empty:
            return df
        df["lat_bin"] = np.round(df["lat"] / coarse) * coarse
        df["lon_bin"] = np.round(df["lon"] / coarse) * coarse

        agg_dict = {}
        for v in vars_keep:
            if v in df.columns:
                series = pd.to_numeric(df[v], errors="coerce")
                agg_dict[f"{v}_mean"] = (v, "mean")
                agg_dict[f"{v}_std"] = (v, "std")
                agg_dict[f"{v}_mean_abs"] = (v, lambda x: np.nanmean(np.abs(pd.to_numeric(x, errors='coerce'))))
        agg = (
            df.groupby(["time", "lat_bin", "lon_bin"], sort=False)
            .agg(**agg_dict, count=("time", "size"))
            .reset_index()
        )
        return agg

    for chunk in _read_iter(args.panel, columns=list(needed), chunk_rows=chunk_rows):
        if chunk.empty:
            continue
        agg_chunk = process(chunk)
        if agg_chunk.empty:
            continue
        write_out(out_path, agg_chunk, writer_state)

    if writer_state.get("writer"):
        writer_state["writer"].close()
    print(f"[oam-meta] wrote meta-field -> {out_path}")


if __name__ == "__main__":
    main()
