#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_gse_states.py

Discretise G/S/E features into coarse levels and attach simple state codes
for downstream tracking/diagnostics.

Outputs the input panel with added columns:
  - G_level, S_level, E_level  (int levels 0..n-1)
  - GSE_str                    (e.g., G2S1E3)
  - slow_phase_bin             (0..n_phase-1) if slow_cos/sin available
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # optional; chunked parquet requires pyarrow
    pa = None
    pq = None


def _is_parquet(path: str) -> bool:
    low = path.lower()
    return low.endswith((".parquet", ".parq", ".pq"))


def _read_any(path: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
    if _is_parquet(path):
        return pd.read_parquet(path, columns=list(columns) if columns else None)
    return pd.read_csv(path, usecols=list(columns) if columns else None, low_memory=False)


def _write_any(path: str, df: pd.DataFrame) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if _is_parquet(str(p)):
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False, date_format="%Y-%m-%d %H:%M:%S")


def _digitize_levels(series: pd.Series, quantiles: Sequence[float]) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    bins = np.quantile(vals.dropna(), quantiles)
    # deduplicate bins to avoid warnings
    bins = np.unique(bins)
    if len(bins) < 2:
        return pd.Series(np.nan, index=series.index, dtype=float)
    # np.digitize returns 1..len(bins); shift to 0-based levels
    levels = np.digitize(vals, bins[1:-1], right=False)
    return levels.astype("Int64")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute discrete G/S/E states and optional slow phase bins.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--panel", required=True, help="Input panel (CSV/Parquet).")
    ap.add_argument("--out", required=True, help="Output panel with GSE states added.")
    ap.add_argument("--g-col", default="G_struct", help="Column for geometry/structure (G).")
    ap.add_argument("--s-col", default="S_shear", help="Column for shear/suppression (S).")
    ap.add_argument("--e-col", default="E_energy", help="Column for energy/thermo (E).")
    ap.add_argument(
        "--quantiles",
        default="0,0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated quantiles for bin edges (e.g., 0,0.2,...,1).",
    )
    ap.add_argument(
        "--slow-cos-col",
        default="slow_cos",
        help="Slow-tick cosine column (optional; ignored if missing).",
    )
    ap.add_argument(
        "--slow-sin-col",
        default="slow_sin",
        help="Slow-tick sine column (optional; ignored if missing).",
    )
    ap.add_argument(
        "--slow-phase-bins",
        type=int,
        default=8,
        help="Number of bins for slow-phase discretisation (if slow cols present).",
    )
    # Chunking hints (accepted for pipeline compatibility)
    ap.add_argument("--chunk-rows", type=int, default=None, help="Optional chunk size for streaming.")
    ap.add_argument("--chunksize", type=int, default=None, help="Alias for chunk-rows.")
    ap.add_argument("--parquet-rows", type=int, default=None, help="Ignored; accepted for compatibility.")
    args = ap.parse_args()

    quantiles = [float(q) for q in args.quantiles.split(",") if q.strip() != ""]
    if quantiles[0] != 0.0 or quantiles[-1] != 1.0:
        raise SystemExit("Quantiles must start at 0 and end at 1.")

    chunk_rows = args.chunk_rows or args.chunksize

    def attach_states(df: pd.DataFrame, bins_g, bins_s, bins_e) -> pd.DataFrame:
        df = df.copy()
        df["G_level"] = _digitize_levels(df.get(args.g_col, pd.Series(dtype=float)), bins_g)
        df["S_level"] = _digitize_levels(df.get(args.s_col, pd.Series(dtype=float)), bins_s)
        df["E_level"] = _digitize_levels(df.get(args.e_col, pd.Series(dtype=float)), bins_e)
        df["GSE_str"] = (
            "G" + df["G_level"].astype(str) + "S" + df["S_level"].astype(str) + "E" + df["E_level"].astype(str)
        )
        if args.slow_cos_col in df.columns and args.slow_sin_col in df.columns:
            phase = np.arctan2(pd.to_numeric(df[args.slow_sin_col], errors="coerce"),
                               pd.to_numeric(df[args.slow_cos_col], errors="coerce"))
            bins_phase = np.linspace(-np.pi, np.pi, args.slow_phase_bins + 1)
            df["slow_phase_bin"] = pd.cut(phase, bins=bins_phase, labels=False, include_lowest=True)
        return df

    if not chunk_rows:
        # simple full-table path
        df = _read_any(args.panel)
        df = attach_states(df, quantiles, quantiles, quantiles)
        _write_any(args.out, df)
        print(f"[gse-states] wrote {len(df):,} rows -> {args.out}")
        return

    # chunked path with approximate quantiles from sampled chunks
    print(f"[gse-states] chunking with chunk_rows={chunk_rows} (quantiles estimated from samples)")
    pstr = str(args.panel).lower()
    is_parquet = pstr.endswith((".parquet", ".parq", ".pq"))
    if is_parquet and pq is None:
        print("[gse-states] pyarrow not available; falling back to full read.")
        df = _read_any(args.panel)
        df = attach_states(df, quantiles, quantiles, quantiles)
        _write_any(args.out, df)
        print(f"[gse-states] wrote {len(df):,} rows -> {args.out}")
        return

    # Pass 1: sample values to estimate quantile bins
    sample_limit = 1_000_000
    rng = np.random.default_rng(42)
    samples = {col: [] for col in (args.g_col, args.s_col, args.e_col)}

    def extend_samples(arr: np.ndarray, key: str):
        arr = pd.to_numeric(pd.Series(arr), errors="coerce").dropna().to_numpy()
        if arr.size == 0:
            return
        take = min(arr.size, max(1, sample_limit // 10))
        choice = rng.choice(arr, size=take, replace=False) if arr.size > take else arr
        buf = samples[key]
        buf.append(choice)
        # truncate if too large
        total = sum(len(x) for x in buf)
        if total > sample_limit:
            merged = np.concatenate(buf)
            buf.clear()
            buf.append(rng.choice(merged, size=sample_limit, replace=False))

    if is_parquet:
        pf = pq.ParquetFile(args.panel)
        for batch in pf.iter_batches(batch_size=chunk_rows, columns=[args.g_col, args.s_col, args.e_col]):
            tbl = batch.to_pandas()
            extend_samples(tbl[args.g_col].to_numpy(), args.g_col)
            extend_samples(tbl[args.s_col].to_numpy(), args.s_col)
            extend_samples(tbl[args.e_col].to_numpy(), args.e_col)
    else:
        for chunk in pd.read_csv(args.panel, chunksize=chunk_rows, low_memory=False):
            extend_samples(chunk.get(args.g_col, []), args.g_col)
            extend_samples(chunk.get(args.s_col, []), args.s_col)
            extend_samples(chunk.get(args.e_col, []), args.e_col)

    def build_bins(key: str):
        arrs = samples[key]
        if not arrs:
            return quantiles
        arr = np.concatenate(arrs)
        return np.quantile(arr, quantiles)

    bins_g = build_bins(args.g_col)
    bins_s = build_bins(args.s_col)
    bins_e = build_bins(args.e_col)
    print(f"[gse-states] bins G:{bins_g} S:{bins_s} E:{bins_e}")

    # Pass 2: stream, transform, and write
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in (".parquet", ".pq", ".parq"):
        writer = None
        if is_parquet:
            pf = pq.ParquetFile(args.panel)
            for batch in pf.iter_batches(batch_size=chunk_rows):
                dfc = batch.to_pandas()
                dfc = attach_states(dfc, bins_g, bins_s, bins_e)
                tbl = pa.Table.from_pandas(dfc, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, tbl.schema)
                writer.write_table(tbl)
        else:
            for chunk in pd.read_csv(args.panel, chunksize=chunk_rows, low_memory=False):
                dfc = attach_states(chunk, bins_g, bins_s, bins_e)
                tbl = pa.Table.from_pandas(dfc, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, tbl.schema)
                writer.write_table(tbl)
        if writer:
            writer.close()
    else:
        first = True
        if is_parquet:
            pf = pq.ParquetFile(args.panel)
            for batch in pf.iter_batches(batch_size=chunk_rows):
                dfc = attach_states(batch.to_pandas(), bins_g, bins_s, bins_e)
                dfc.to_csv(out_path, mode="w" if first else "a", index=False, header=first)
                first = False
        else:
            for chunk in pd.read_csv(args.panel, chunksize=chunk_rows, low_memory=False):
                dfc = attach_states(chunk, bins_g, bins_s, bins_e)
                dfc.to_csv(out_path, mode="w" if first else "a", index=False, header=first)
                first = False

    print(f"[gse-states] wrote (chunked) -> {args.out}")


if __name__ == "__main__":
    main()
