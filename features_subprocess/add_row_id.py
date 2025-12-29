#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_row_id.py

Add stable row_id = hash(time_floor_H, lat_round3, lon_round3) to a Parquet
table. Intended to run immediately after labelling so row_id is preserved
through subsequent feature stages.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.util import hash_pandas_object

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pa = None
    pq = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Add row_id (hash of time/lat/lon) to a labelled Parquet file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--infile", required=True, help="Input Parquet with time/lat/lon (and labels).")
    ap.add_argument("--outfile", required=True, help="Output Parquet with row_id added.")
    ap.add_argument(
        "--chunk-rows",
        "--chunksize",
        "--parquet-rows",
        "--parquet_rows",
        type=int,
        default=200_000,
        dest="chunk_rows",
        help="Parquet batch size for streaming.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outfile.")
    ap.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip work if outfile already exists.",
    )
    return ap.parse_args()


def _require_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns {missing}; got {list(df.columns)[:10]} ...")


def main() -> None:
    args = parse_args()
    src = Path(args.infile)
    out = Path(args.outfile)

    if not src.exists():
        raise SystemExit(f"Input file not found: {src}")
    if out.exists():
        if args.skip_if_exists:
            print(f"[add-row-id] skip (exists): {out}")
            return
        if not args.overwrite:
            print(f"[add-row-id] skip (exists, use --overwrite or --skip-if-exists): {out}")
            return

    if pq is None or pa is None:
        raise SystemExit("pyarrow is required for Parquet I/O; install pyarrow.")

    pf = pq.ParquetFile(src)
    writer = None
    total = 0
    for i, batch in enumerate(
        pf.iter_batches(batch_size=int(args.chunk_rows) if args.chunk_rows else None),
        start=1,
    ):
        df = batch.to_pandas()
        _require_cols(df, ["time", "lat", "lon"])
        t = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_convert(None).dt.floor("H")
        la = pd.to_numeric(df["lat"], errors="coerce").round(3)
        lo = pd.to_numeric(df["lon"], errors="coerce").round(3)
        key_frame = pd.DataFrame({"time": t, "lat": la, "lon": lo})
        row_id = hash_pandas_object(key_frame, index=False).astype("uint64")
        df.insert(0, "row_id", row_id.values)

        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            out.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(out, table.schema, compression="snappy")
        writer.write_table(table)
        total += len(df)
        print(f"[add-row-id] chunk {i}: wrote {len(df):,} rows (cum={total:,})")

    if writer is not None:
        writer.close()
    print(f"[add-row-id] done -> {out} rows={total:,}")


if __name__ == "__main__":
    main()
