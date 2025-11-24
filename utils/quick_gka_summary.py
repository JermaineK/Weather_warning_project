#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quick_gka_summary.py — lightweight inspection for large GKA parquet files.

- Uses Parquet metadata for size/row-group/row counts.
- Reads per-column null counts and min/max hints when available (from row-group stats).
- Samples a small slice of gka_* columns for quick quantiles without loading everything.

Usage:
  python utils/quick_gka_summary.py data/grid_labelled_FMA_gka.parquet
  python utils/quick_gka_summary.py data/grid_labelled_FMA_gka.parquet --sample-rows 300000
  python utils/quick_gka_summary.py data/grid_labelled_FMA_gka.parquet --prefix gka_
"""

from __future__ import annotations
import sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

def _to_scalar(v):
    try:
        if isinstance(v, (bytes, bytearray)):
            return v.decode("utf-8", errors="replace")
        return v.as_py() if hasattr(v, "as_py") else v
    except Exception:
        return v

def summarize_parquet(path: str, sample_rows: int = 200_000, gka_prefix: str = "gka_"):
    p = Path(path)
    if not p.exists():
        print(f"[err] File not found: {p}")
        sys.exit(1)

    pf = pq.ParquetFile(p)

    # ----- high-level facts
    schema_names = list(pf.schema.names)
    num_cols = len(schema_names)
    num_row_groups = pf.num_row_groups
    total_rows = pf.metadata.num_rows if pf.metadata is not None else sum(
        pf.metadata.row_group(i).num_rows for i in range(num_row_groups)
    )
    fsize_gb = p.stat().st_size / 1e9

    # Arrow schema for dtypes (ParquetFile.schema_arrow is the Arrow schema)
    arrow_schema = getattr(pf, "schema_arrow", None)
    have_arrow_schema = isinstance(arrow_schema, pa.Schema)

    print(f"\n📦 {p.name}")
    print(f"  row groups: {num_row_groups:,}")
    print(f"  columns:    {num_cols:,}")
    print(f"  rows:       {total_rows:,}")
    print(f"  size:       {fsize_gb:.2f} GB\n")

    # ----- per-column quick stats from row-group metadata
    rows = []
    for col_idx, name in enumerate(schema_names):
        # dtype: prefer Arrow schema (stable); fallback to Parquet physical if needed
        if have_arrow_schema:
            try:
                # field() accepts either name or index
                dtype = str(arrow_schema.field(name).type)
            except Exception:
                # fallback by index
                try:
                    dtype = str(arrow_schema.field(col_idx).type)
                except Exception:
                    dtype = "unknown"
        else:
            dtype = "unknown"

        nulls_total = 0
        have_stats_any = False
        min_val = None
        max_val = None

        for rg in range(num_row_groups):
            col_meta = pf.metadata.row_group(rg).column(col_idx)
            stats = col_meta.statistics
            if stats is None:
                continue
            have_stats_any = True
            if getattr(stats, "has_null_count", False):
                nulls_total += stats.null_count
            if getattr(stats, "has_min_max", False) and min_val is None:
                try:
                    min_val = _to_scalar(stats.min)
                    max_val = _to_scalar(stats.max)
                except Exception:
                    pass

        rows.append((name, dtype, min_val, max_val, (nulls_total if have_stats_any else None)))

    df_meta = pd.DataFrame(rows, columns=["column", "dtype", "min_hint", "max_hint", "nulls(sum)"])
    with pd.option_context("display.width", 160, "display.max_rows", 60):
        print(df_meta.head(40).to_string(index=False))

    # ----- sample gka_* columns for quantiles
    gka_cols = [c for c in schema_names if c.startswith(gka_prefix)]
    if not gka_cols:
        print(f"\n(no columns starting with '{gka_prefix}' detected)")
        return

    # 1) cheap head sample via pandas (respects nrows)
    head_n = max(50_000, min(sample_rows, 200_000))
    try:
        head_sample = pd.read_parquet(p, columns=gka_cols, engine="pyarrow", nrows=head_n)
    except Exception as e:
        print(f"\n[warn] Unable to read head sample: {e}")
        head_sample = pd.DataFrame(columns=gka_cols)

    # 2) add one mid-file sample via row-group read (avoids only-head bias)
    tail_sample = None
    try:
        mid_rg = min(max(num_row_groups // 3, 0), max(num_row_groups - 1, 0))
        if num_row_groups > 0:
            # Prefer read_row_group (stable); fallback to iter_batches with row_groups arg
            try:
                ta = pf.read_row_group(mid_rg, columns=gka_cols)
                tail_sample = ta.to_pandas()
                # reduce if it's huge
                if len(tail_sample) > head_n:
                    tail_sample = tail_sample.sample(n=head_n, random_state=123, ignore_index=True)
            except Exception:
                # Older pyarrow fallback
                batch_iter = pf.iter_batches(batch_size=min(sample_rows, 200_000), columns=gka_cols)
                tail_sample = next(batch_iter).to_pandas()
    except Exception as e:
        print(f"[warn] Mid-file sample failed: {e}")

    if tail_sample is not None and not tail_sample.empty:
        sample_df = pd.concat([head_sample, tail_sample], ignore_index=True)
    else:
        sample_df = head_sample

    if sample_df.empty:
        print("\n(no sample could be loaded; columns may be non-numeric or file is inaccessible)")
        return

    numeric = sample_df.select_dtypes(include=["number"])
    if numeric.empty:
        print("\n(gka_* columns are non-numeric in sample; skipping quantiles)")
        return

    desc = numeric.describe(percentiles=[0.05, 0.5, 0.95]).T
    miss = numeric.isna().mean() * 100.0
    desc["missing_%"] = miss

    print("\n== GKA Feature Quantiles (sample) ==")
    with pd.option_context("display.width", 160, "display.max_rows", 80):
        print(desc[["min", "5%", "50%", "95%", "max", "missing_%"]])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python utils/quick_gka_summary.py <file.parquet> [--sample-rows N] [--prefix gka_]")
        sys.exit(1)

    path = sys.argv[1]
    sample = 200_000
    prefix = "gka_"
    if "--sample-rows" in sys.argv:
        i = sys.argv.index("--sample-rows")
        if i + 1 < len(sys.argv):
            try:
                sample = int(sys.argv[i + 1])
            except Exception:
                pass
    if "--prefix" in sys.argv:
        i = sys.argv.index("--prefix")
        if i + 1 < len(sys.argv):
            prefix = sys.argv[i + 1]

    summarize_parquet(path, sample_rows=sample, gka_prefix=prefix)