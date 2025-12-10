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
    args = ap.parse_args()

    quantiles = [float(q) for q in args.quantiles.split(",") if q.strip() != ""]
    if quantiles[0] != 0.0 or quantiles[-1] != 1.0:
        raise SystemExit("Quantiles must start at 0 and end at 1.")

    df = _read_any(args.panel)

    # G/S/E levels
    df["G_level"] = _digitize_levels(df.get(args.g_col, pd.Series(dtype=float)), quantiles)
    df["S_level"] = _digitize_levels(df.get(args.s_col, pd.Series(dtype=float)), quantiles)
    df["E_level"] = _digitize_levels(df.get(args.e_col, pd.Series(dtype=float)), quantiles)

    df["GSE_str"] = (
        "G" + df["G_level"].astype(str) + "S" + df["S_level"].astype(str) + "E" + df["E_level"].astype(str)
    )

    # Slow phase (optional)
    slow_cols_present = args.slow_cos_col in df.columns and args.slow_sin_col in df.columns
    if slow_cols_present:
        phase = np.arctan2(pd.to_numeric(df[args.slow_sin_col], errors="coerce"),
                           pd.to_numeric(df[args.slow_cos_col], errors="coerce"))
        bins = np.linspace(-np.pi, np.pi, args.slow_phase_bins + 1)
        df["slow_phase_bin"] = pd.cut(phase, bins=bins, labels=False, include_lowest=True)

    _write_any(args.out, df)
    print(f"[gse-states] wrote {len(df):,} rows -> {args.out}")


if __name__ == "__main__":
    main()
