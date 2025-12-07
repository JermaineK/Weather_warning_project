#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_slowtick_features.py

Add simple slow-tick oscillatory features to a (lagged) GSE panel so the model
can learn phase effects (e.g., 24–30h modes) instead of only relying on post-hoc diagnostics.

Assumes input is already a downselected training subset (memory-safe).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _is_parquet(path: str) -> bool:
    low = path.lower()
    return low.endswith((".parquet", ".parq", ".pq"))


def _read_any(path: str) -> pd.DataFrame:
    if _is_parquet(path):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _write_any(path: str, df: pd.DataFrame) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if _is_parquet(str(p)):
        df.to_parquet(p, index=False)
        return
    comp = "gzip" if p.name.lower().endswith(".gz") else "infer"
    df.to_csv(p, index=False, compression=comp, date_format="%Y-%m-%d %H:%M:%S")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add slow-tick cosine/sine features (and optional modulated variants) to a GSE panel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--panel", required=True, help="Input GSE panel (CSV(.gz) or Parquet).")
    ap.add_argument("--out", required=True, help="Output path with slow-tick features.")
    ap.add_argument("--time-col", default="time", help="Timestamp column (UTC naive).")
    ap.add_argument("--period-hours", type=float, default=24.0, help="Slow-tick period in hours.")
    ap.add_argument(
        "--use-modulated",
        action="store_true",
        help="Also add G_struct-modulated slow features (G_slow_cos/sin).",
    )
    ap.add_argument(
        "--chunksize",
        "--chunk-rows",
        "--chunk_rows",
        "--parquet-rows",
        "--parquet_rows",
        type=int,
        default=None,
        help="Optional chunk size for streaming CSV/Parquet input (0/None = load whole file).",
    )
    args = ap.parse_args()

    chunk_rows = args.chunksize if args.chunksize and args.chunksize > 0 else None

    def process_chunk(df: pd.DataFrame, t0: pd.Timestamp | None) -> tuple[pd.DataFrame, pd.Timestamp | None]:
        if args.time_col not in df.columns:
            raise SystemExit(f"Missing time column '{args.time_col}' in {args.panel}")
        t = pd.to_datetime(df[args.time_col], utc=True, errors="coerce").dt.tz_convert(None)
        if t.isna().all():
            raise SystemExit(f"Could not parse any times from column '{args.time_col}'")
        if t0 is None:
            t0 = t.min()
        dt_h = (t - t0) / np.timedelta64(1, "h")

        omega = 2 * np.pi / float(args.period_hours)
        slow_cos = np.cos(omega * dt_h.to_numpy())
        slow_sin = np.sin(omega * dt_h.to_numpy())

        df["slow_cos"] = slow_cos.astype("float32")
        df["slow_sin"] = slow_sin.astype("float32")

        if args.use_modulated:
            if "G_struct" in df.columns:
                g = df["G_struct"].to_numpy(dtype=float)
                df["G_slow_cos"] = (g * slow_cos).astype("float32")
                df["G_slow_sin"] = (g * slow_sin).astype("float32")
            else:
                print("[warn] G_struct not found; skipping modulated slow-tick features.")
        return df, t0

    # streaming read/write
    is_parquet = _is_parquet(args.panel)
    writer = None
    total = 0
    t0_global = None

    if chunk_rows:
        if is_parquet:
            import pyarrow.parquet as pq  # type: ignore
            pf = pq.ParquetFile(args.panel)
            iterator = (batch.to_pandas() for batch in pf.iter_batches(batch_size=chunk_rows))
        else:
            iterator = pd.read_csv(args.panel, low_memory=False, chunksize=chunk_rows)
    else:
        iterator = [_read_any(args.panel)]

    for ch in iterator:
        df_chunk = ch
        df_chunk, t0_global = process_chunk(df_chunk, t0_global)
        if writer is None:
            # init output based on first chunk type
            writer = _write_any
        _write_any(args.out, df_chunk if writer is _write_any else df_chunk)
        total += len(df_chunk)

    print(f"[done] wrote slow-tick features -> {args.out}  (rows={total:,})")


if __name__ == "__main__":
    main()
