#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_storm_timeseries_by_id.py

Storm-centred G/S/E time series aggregation (stub).

For now this script acts as a streaming passthrough/copy that accepts the
pipeline’s chunking flags so the stage can run end-to-end without errors.
When the real aggregation is implemented, replace the passthrough block
with the proper logic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pa = None
    pq = None


# Agent: stub only. Implement aggregation when slow-tick analysis is ready.


def main() -> None:
    ap = argparse.ArgumentParser(
        description="(Stub) Build storm-centred time series of G/S/E fields around track points.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--labelled-with-id", help="Rich grid with row_id and features.", required=False)
    ap.add_argument("--tracks", help="Storm tracks table (CSV/Parquet).", required=False)
    ap.add_argument("--out", help="Output time series panel.", required=False)
    ap.add_argument("--t-before", type=float, default=240.0, help="Hours before reference time.")
    ap.add_argument("--t-after", type=float, default=48.0, help="Hours after reference time.")
    ap.add_argument("--radius-deg", type=float, default=2.0, help="Radius (deg) around track point.")
    ap.add_argument("--ref-kind", default="genesis", help="Reference time kind: genesis|max-int.")
    # Accept chunking hints for pipeline compatibility (ignored in stub).
    ap.add_argument("--chunk-rows", type=int, default=None, help="Ignored; accepted for pipeline compatibility.")
    ap.add_argument("--chunksize", type=int, default=None, help="Alias for --chunk-rows (ignored).")
    ap.add_argument("--parquet-rows", type=int, default=None, help="Ignored; accepted for pipeline compatibility.")
    args = ap.parse_args()

    # Minimal passthrough so the pipeline stage completes.
    if not args.labelled_with_id or not args.out:
        print(
            "[stub] build_storm_timeseries_by_id.py is not implemented yet. "
            "Provide --labelled-with-id/--out to run a streaming copy."
        )
        sys.exit(0)

    src = Path(args.labelled_with_id)
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    chunk_rows = args.chunk_rows or args.chunksize or 0
    parquet_rows = args.parquet_rows or chunk_rows or 0

    suffix = src.suffix.lower()
    is_parquet = suffix in {".parquet", ".parq", ".pq"}

    if is_parquet and pq is not None:
        pf = pq.ParquetFile(src)
        writer = None
        rows = 0
        for batch in pf.iter_batches(batch_size=parquet_rows or 200_000):
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(dst, table.schema, compression="snappy")
            writer.write_table(table)
            rows += table.num_rows
        if writer is not None:
            writer.close()
        print(f"[stub] passthrough copy parquet -> {dst} rows={rows:,}")
    else:
        # CSV/GZ fallback
        comp = "gzip" if str(dst).lower().endswith(".gz") else "infer"
        mode = "w"
        header = True
        rows = 0
        for chunk in pd.read_csv(
            src,
            compression="infer",
            low_memory=False,
            encoding_errors="replace",
            on_bad_lines="skip",
            chunksize=chunk_rows or 200_000,
            parse_dates=["time"],
        ):
            chunk.to_csv(
                dst,
                index=False,
                mode=mode,
                header=header,
                compression=comp,
                date_format="%Y-%m-%d %H:%M:%S",
            )
            mode = "a"
            header = False
            rows += len(chunk)
        print(f"[stub] passthrough copy csv -> {dst} rows={rows:,}")


if __name__ == "__main__":
    main()
