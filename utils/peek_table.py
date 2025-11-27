#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
peek_table.py — column + sample peek for CSV(.gz)/Parquet.
"""

import argparse
import pandas as pd
from pathlib import Path

from utils import io_common

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="CSV/CSV.GZ or Parquet")
    ap.add_argument("--nrows", type=int, default=10)
    args = ap.parse_args()

    p = Path(args.path)
    low = p.name.lower()
    if low.endswith((".parquet",".parq",".pq")):
        df = pd.read_parquet(p)
    else:
        df = io_common.read_any(
            p,
            nrows=max(args.nrows, 200_000),
            compression="infer",
        )

    cols = list(df.columns)
    print(f"\n{p} — columns ({len(cols)}):")
    print(cols)

    print(f"\nSample (n={min(len(df), args.nrows)}):")
    with pd.option_context("display.max_columns", 200, "display.width", 160):
        print(df.head(args.nrows))

if __name__ == "__main__":
    main()