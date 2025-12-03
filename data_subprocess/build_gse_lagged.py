#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_gse_lagged.py

Add temporal persistence features to a GSE panel (post-subset) to capture drift/persistence.

Outputs same-cell lag features and optional "nearby previous-hour" maxima so moving structure
still shows up even if it hopped a grid box.

Assumes the input already fits in memory (ID subset). If your subset is huge, downsample first.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

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


def _parse_lags(spec: str) -> List[int]:
    if not spec:
        return [1]
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    lags = []
    for p in parts:
        try:
            lags.append(int(p))
        except Exception:
            continue
    lags = [l for l in lags if l > 0]
    return lags or [1]


def _max_within_radius(prev_block: pd.DataFrame, lat_now: np.ndarray, lon_now: np.ndarray, col: str, r: float) -> np.ndarray:
    """
    For each row in the current block, compute max of `col` in prev_block within |dlat|,|dlon| <= r.
    Uses a simple bounding-box filter (degrees). Assumes prev_block has columns ["lat_r","lon_r",col].
    """
    if prev_block.empty:
        return np.full_like(lat_now, np.nan, dtype=float)
    la_prev = prev_block["lat_r"].to_numpy()
    lo_prev = prev_block["lon_r"].to_numpy()
    vals = prev_block[col].to_numpy()
    out = np.full_like(lat_now, np.nan, dtype=float)

    for i, (la0, lo0) in enumerate(zip(lat_now, lon_now)):
        m = (np.abs(la_prev - la0) <= r) & (np.abs(lo_prev - lo0) <= r)
        if m.any():
            out[i] = np.nanmax(vals[m])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add temporal lag features (same-cell and optional nearby drift) to a GSE panel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--panel", required=True, help="Input GSE panel (CSV(.gz) or Parquet).")
    ap.add_argument("--out", required=True, help="Output with lagged features.")
    ap.add_argument("--lags", default="1", help="Comma-separated hour lags (e.g., '1,2').")
    ap.add_argument("--radius-deg", type=float, default=0.0, help="Radius (deg) for drift-aware prev-hour max (0 disables).")
    ap.add_argument("--lat-col", default="lat", help="Latitude column.")
    ap.add_argument("--lon-col", default="lon", help="Longitude column.")
    ap.add_argument("--time-col", default="time", help="Timestamp column.")
    ap.add_argument("--round-dp", type=int, default=3, help="Decimal places to round lat/lon for same-cell grouping.")
    args = ap.parse_args()

    lags = _parse_lags(args.lags)
    df = _read_any(args.panel)

    if args.time_col not in df or args.lat_col not in df or args.lon_col not in df:
        missing = [c for c in (args.time_col, args.lat_col, args.lon_col) if c not in df]
        raise SystemExit(f"Missing required columns: {missing}")

    df[args.time_col] = pd.to_datetime(df[args.time_col], utc=True, errors="coerce").dt.tz_convert(None)
    df["lat_r"] = pd.to_numeric(df[args.lat_col], errors="coerce").round(args.round_dp)
    df["lon_r"] = pd.to_numeric(df[args.lon_col], errors="coerce").round(args.round_dp)
    df = df.sort_values(["lat_r", "lon_r", args.time_col])

    # same-cell lags
    grp = df.groupby(["lat_r", "lon_r"], sort=False)
    for lag in lags:
        for src, tgt in (("G_struct", f"G_prev{lag}h_same"),
                         ("S_shear", f"S_prev{lag}h_same"),
                         ("E_energy", f"E_prev{lag}h_same")):
            if src in df:
                df[tgt] = grp[src].shift(lag)

    # drift-aware lags (radius > 0)
    r = float(args.radius_deg)
    if r > 0:
        df["time_floor"] = df[args.time_col].dt.floor("H")
        time_groups = {t: block[["lat_r", "lon_r", "G_struct", "S_shear", "E_energy"]] for t, block in df.groupby("time_floor", sort=True)}

        for lag in lags:
            col_suffix = f"prev{lag}h_near"
            for src, tgt in (("G_struct", f"G_{col_suffix}"),
                             ("S_shear", f"S_{col_suffix}"),
                             ("E_energy", f"E_{col_suffix}")):
                df[tgt] = np.nan
            for t, block_now in df.groupby("time_floor", sort=True):
                prev_t = t - pd.Timedelta(hours=lag)
                prev_block = time_groups.get(prev_t)
                if prev_block is None or prev_block.empty:
                    continue
                la_now = block_now["lat_r"].to_numpy()
                lo_now = block_now["lon_r"].to_numpy()
                for src, tgt in (("G_struct", f"G_{col_suffix}"),
                                 ("S_shear", f"S_{col_suffix}"),
                                 ("E_energy", f"E_{col_suffix}")):
                    if src in prev_block:
                        vals = _max_within_radius(prev_block[["lat_r", "lon_r", src]], la_now, lo_now, src, r)
                        df.loc[block_now.index, tgt] = vals

        df.drop(columns=["time_floor"], inplace=True)

    # cleanup
    df.drop(columns=["lat_r", "lon_r"], inplace=True)
    _write_any(args.out, df)
    print(f"[done] wrote {len(df):,} rows with lags -> {args.out}")


if __name__ == "__main__":
    main()
