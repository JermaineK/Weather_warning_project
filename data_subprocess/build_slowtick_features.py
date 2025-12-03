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
    args = ap.parse_args()

    df = _read_any(args.panel)
    if args.time_col not in df.columns:
        raise SystemExit(f"Missing time column '{args.time_col}' in {args.panel}")

    t = pd.to_datetime(df[args.time_col], utc=True, errors="coerce").dt.tz_convert(None)
    if t.isna().all():
        raise SystemExit(f"Could not parse any times from column '{args.time_col}'")
    t0 = t.min()
    dt_h = (t - t0) / np.timedelta64(1, "h")

    omega = 2 * np.pi / float(args.period_hours)
    slow_cos = np.cos(omega * dt_h.to_numpy())
    slow_sin = np.sin(omega * dt_h.to_numpy())

    df["slow_cos"] = slow_cos.astype("float32")
    df["slow_sin"] = slow_sin.astype("float32")

    if args.use_modulated:
        if "G_struct" in df.columns:
            df["G_slow_cos"] = (df["G_struct"].to_numpy(dtype=float) * slow_cos).astype("float32")
            df["G_slow_sin"] = (df["G_struct"].to_numpy(dtype=float) * slow_sin).astype("float32")
        else:
            print("[warn] G_struct not found; skipping modulated slow-tick features.")

    _write_any(args.out, df)
    print(f"[done] wrote slow-tick features -> {args.out}  (rows={len(df):,})")


if __name__ == "__main__":
    main()
