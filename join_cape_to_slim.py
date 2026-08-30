#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
join_cape_to_slim.py — attach real ERA5 CAPE/CIN to a per-season slim panel.

Why this instead of a feature rebuild
    The genesis analyses only need CAPE aligned to cells that already exist in
    data/genesis_<year>_slim.parquet, and those files already carry
    (time, lat, lon) on the native ERA5 grid. Joining directly avoids re-running
    the ~112 GB feature chain just to add two columns.

Tests the last untested leg of the G/S/E hypothesis: `E_energy` in the existing
panels is a msl/t2m-derived proxy (thermo_shear + pdrop_nd + t2m_anom_local),
not convective energy, and it showed no discrimination (AUC 0.510).

USAGE
    python join_cape_to_slim.py --year 2023
    python join_cape_to_slim.py --year 2023 --out data/genesis_2023_slim_cape.parquet
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def month_frames(nc_paths, wanted_times: pd.DatetimeIndex):
    """Yield tidy (time, lat, lon, cape, cin) frames, one per file, restricted
    to timestamps that actually appear in the panel."""
    import xarray as xr
    for p in nc_paths:
        ds = xr.open_dataset(p)
        tname = "valid_time" if "valid_time" in ds.coords else "time"
        latn = "latitude" if "latitude" in ds.coords else "lat"
        lonn = "longitude" if "longitude" in ds.coords else "lon"
        keep = pd.DatetimeIndex(pd.to_datetime(ds[tname].values))
        sel = keep.isin(wanted_times)
        if not sel.any():
            ds.close()
            continue
        ds = ds.isel({tname: np.where(sel)[0]})
        df = ds.to_dataframe().reset_index()
        ds.close()
        df = df.rename(columns={tname: "time", latn: "lat", lonn: "lon"})
        cols = ["time", "lat", "lon"] + [c for c in ("cape", "cin") if c in df.columns]
        df = df[cols]
        df["time"] = pd.to_datetime(df["time"])
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce").round(2)
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce").round(2)
        print(f"[cape] {Path(p).name}: {len(df):,} grid-hours")
        yield df


def main() -> int:
    a = parse_args()
    src = Path(a.panel or f"data/genesis_{a.year}_slim.parquet")
    if not src.exists():
        raise SystemExit(f"[cape] panel not found: {src}")
    out = Path(a.out or str(src).replace("_slim.parquet", "_slim_cape.parquet"))

    nc = sorted(glob.glob(a.nc_glob or f"data_era5/extracted/{a.year}/**/era5_{a.year}*_cape.nc",
                          recursive=True))
    if not nc:
        raise SystemExit(f"[cape] no CAPE files matched for {a.year}")
    print(f"[cape] {len(nc)} CAPE file(s) for {a.year}")

    panel = pd.read_parquet(src)
    panel["time"] = pd.to_datetime(panel["time"])
    panel["lat"] = pd.to_numeric(panel["lat"], errors="coerce").round(2)
    panel["lon"] = pd.to_numeric(panel["lon"], errors="coerce").round(2)
    print(f"[cape] panel {src.name}: {len(panel):,} rows")

    wanted = pd.DatetimeIndex(panel["time"].unique())
    add = pd.concat(list(month_frames(nc, wanted)), ignore_index=True)
    add = add.drop_duplicates(subset=["time", "lat", "lon"])
    print(f"[cape] lookup rows: {len(add):,}")

    before = len(panel)
    merged = panel.merge(add, on=["time", "lat", "lon"], how="left")
    if len(merged) != before:
        raise SystemExit(f"[cape] row count changed on join ({before:,} -> {len(merged):,}); "
                         "duplicate keys in the CAPE lookup.")
    for c in ("cape", "cin"):
        if c in merged.columns:
            cov = float(merged[c].notna().mean())
            print(f"[cape] {c}: coverage {cov:.1%}  "
                  f"median={merged[c].median():.1f}  max={merged[c].max():.1f}")
            if cov < 0.5:
                print(f"[cape] WARNING: low coverage for {c}; check grid alignment.",
                      file=sys.stderr)

    merged.to_parquet(out, index=False)
    sz = out.stat().st_size / 1073741824
    print(f"[cape] wrote {out}  rows={len(merged):,}  {sz:.2f} GB")
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description="Join ERA5 CAPE/CIN onto a season slim panel.")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--panel", default=None)
    ap.add_argument("--nc-glob", default=None)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        print(f"\nERROR [{type(e).__name__}]: {e}\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)
