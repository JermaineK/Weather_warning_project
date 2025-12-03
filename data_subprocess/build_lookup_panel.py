#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_lookup_panel.py

Purpose:
  - Take an ID-labelled source table and pull a slim panel for a set of IDs
    chosen downstream (e.g., subset selections, diagnostics).
  - Lets you re-run lookups without rebuilding IDs.

Inputs:
  --source   : ID-labelled table (CSV(.gz)/Parquet) containing row_id
  --ids-file : File with IDs to keep (CSV/Parquet), reads --ids-col
  --ids      : Optional inline comma-separated IDs
  --keep-cols: Columns to keep (defaults to all)

Output:
  --out      : Slim panel (CSV(.gz)/Parquet)

Chunking:
  --chunksize / --chunk-rows / --parquet-rows control streaming read.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Set

import numpy as np
import pandas as pd

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pa = None
    pq = None


def _is_parquet(path: str) -> bool:
    low = path.lower()
    return low.endswith((".parquet", ".parq", ".pq"))


def _iter_table(path: str, chunksize: Optional[int]) -> Iterable[pd.DataFrame]:
    if _is_parquet(path):
        if pq is None:
            yield pd.read_parquet(path)
            return
        pf = pq.ParquetFile(path)
        if chunksize and chunksize > 0:
            for batch in pf.iter_batches(batch_size=int(chunksize)):
                yield batch.to_pandas()
            return
        for rg in range(pf.num_row_groups):
            yield pf.read_row_group(rg).to_pandas()
        return

    if chunksize and chunksize > 0:
        for ch in pd.read_csv(path, low_memory=False, chunksize=int(chunksize)):
            yield ch
        return
    yield pd.read_csv(path, low_memory=False)


def _to_python_id(val: object) -> Optional[object]:
    if pd.isna(val):
        return None
    try:
        return int(val)
    except Exception:
        return str(val)


class _Writer:
    def __init__(self, dest: Path):
        self.dest = dest
        self._mode = "parquet" if _is_parquet(str(dest)) else "csv"
        self._header_written = False
        self._pw = None
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        if self.dest.exists():
            print(f"[warn] output {self.dest} exists; overwriting.")
            self.dest.unlink()
        if self._mode == "parquet" and pq is None:
            raise SystemExit("pyarrow is required for parquet output; install pyarrow or choose CSV.")

    def write(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        if self._mode == "parquet":
            assert pa is not None and pq is not None
            tbl = pa.Table.from_pandas(df, preserve_index=False)
            if self._pw is None:
                self._pw = pq.ParquetWriter(self.dest, tbl.schema)
            self._pw.write_table(tbl)
            return
        comp = "gzip" if self.dest.name.lower().endswith(".gz") else "infer"
        df.to_csv(
            self.dest,
            index=False,
            mode="w" if not self._header_written else "a",
            header=not self._header_written,
            compression=comp,
            date_format="%Y-%m-%d %H:%M:%S",
        )
        self._header_written = True

    def close(self) -> None:
        if self._pw is not None:
            self._pw.close()
            self._pw = None


def _load_ids(path: str, col: str) -> Set[object]:
    if not path:
        return set()
    cols = [col]
    if _is_parquet(path):
        df = pd.read_parquet(path, columns=cols)
    else:
        df = pd.read_csv(path, usecols=cols, low_memory=False)
    return {_to_python_id(v) for v in df[col].dropna()}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Lookup slim panel rows for a set of IDs (row_id) without rebuilding IDs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--source", required=True, help="ID-labelled source table (CSV(.gz) or Parquet).")
    ap.add_argument("--out", required=True, help="Output slim panel (CSV(.gz) or Parquet).")
    ap.add_argument("--ids-file", default=None, help="File with IDs to keep (reads --ids-col).")
    ap.add_argument("--ids", default=None, help="Optional comma-separated list of IDs to include.")
    ap.add_argument("--ids-col", default="row_id", help="ID column name in source and ids-file.")
    ap.add_argument(
        "--keep-cols",
        nargs="*",
        default=[],
        help="Columns to keep (row_id always kept). Empty = keep all columns.",
    )
    ap.add_argument(
        "--chunksize",
        "--chunk-rows",
        "--chunk_rows",
        "--parquet-rows",
        "--parquet_rows",
        type=int,
        default=400_000,
        dest="chunksize",
        help="Chunk size for streaming source (CSV rows or parquet batch size).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output file.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        print(f"[lookup] skip (exists, use --overwrite): {out_path}")
        return

    ids: Set[object] = set()
    if args.ids_file:
        ids |= _load_ids(args.ids_file, args.ids_col)
    if args.ids:
        for part in str(args.ids).split(","):
            part = part.strip()
            if part:
                ids.add(_to_python_id(part))

    ids = {i for i in ids if i is not None}
    if not ids:
        raise SystemExit("No IDs provided. Use --ids-file and/or --ids.")

    keep_cols = [c for c in (args.keep_cols or []) if c]
    writer = _Writer(out_path)
    total_rows = 0

    print(
        f"[lookup] source={args.source} ids={len(ids):,} keep_cols="
        f"{keep_cols if keep_cols else '(all)'} chunksize={args.chunksize}"
    )

    for i, chunk in enumerate(_iter_table(args.source, args.chunksize), start=1):
        if chunk is None or chunk.empty:
            continue
        if args.ids_col not in chunk.columns:
            raise SystemExit(f"Chunk {i} missing ids-col '{args.ids_col}'. Columns: {list(chunk.columns)[:10]}")
        id_norm = chunk[args.ids_col].apply(_to_python_id)
        mask = id_norm.isin(ids)
        sub = chunk.loc[mask]
        if keep_cols:
            cols = ["row_id"] if "row_id" in sub.columns else []
            cols += [c for c in keep_cols if c in sub.columns and c not in cols]
            missing = [c for c in keep_cols if c not in sub.columns]
            if missing:
                print(f"[lookup] chunk {i}: missing requested columns {missing}; keeping available ones.")
            sub = sub.loc[:, cols] if cols else sub
        writer.write(sub)
        total_rows += len(sub)
        print(f"[lookup] chunk {i}: kept {len(sub):,} rows (cum={total_rows:,})")

    writer.close()
    if total_rows == 0:
        print("[lookup] wrote 0 rows; check IDs and source alignment.")
    else:
        print(f"[lookup] done -> {args.out} (rows={total_rows:,})")


if __name__ == "__main__":
    main()
