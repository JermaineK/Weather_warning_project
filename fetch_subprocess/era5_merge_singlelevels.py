#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
era5_merge_singlelevels.py
Merge already-downloaded ERA5 single-level NetCDFs per month into a tidy union file.

- Standardizes coords to (time, lat, lon)
- Wraps lon to [-180, 180) and sorts lat ascending
- Aliases common long variable names to short keys (u10, v10, t2m, msl, etc.)
- Keeps only numeric data variables
- Robustly merges files with xr.merge (no open_mfdataset pitfalls)

Usage:
  python era5_merge_singlelevels.py \
      --in-glob "data_era5/extracted/**/era5_*.nc" \
      --out-dir data_era5/extracted \
      --suffix oper \
      --overwrite
"""

from __future__ import annotations
import argparse, glob, re, sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import xarray as xr

# ---- aliases to short names ----
ALIASES: Dict[str, str] = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "total_column_water_vapour": "tcwv",
    "total_precipitation": "tp",
    "surface_latent_heat_flux": "slhf",
    "surface_sensible_heat_flux": "sshf",
}

MONTH_RE = re.compile(r"(19|20)\d{2}(0[1-9]|1[0-2])")  # YYYYMM


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Merge ERA5 single-level NetCDFs per month into union files."
    )
    ap.add_argument(
        "--in-glob",
        required=True,
        help='Glob of input .nc files, e.g. "data_era5/extracted/**/era5_*.nc"',
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Where to write era5_single_YYYYMM_<suffix>.nc",
    )
    ap.add_argument(
        "--suffix",
        default="oper",
        help="Suffix for output filenames (default: oper)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging",
    )
    return ap.parse_args()


def find_month_token(path: Path) -> Optional[str]:
    m = MONTH_RE.search(path.name)
    return m.group(0) if m else None


def wrap_lon_180(lon_da: xr.DataArray) -> xr.DataArray:
    lon = lon_da.astype("float64")
    lon2 = ((lon + 180.0) % 360.0) - 180.0
    return lon2


def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    # rename common alternates
    ren = {}
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ren["valid_time"] = "time"
    if "latitude" in ds.coords:
        ren["latitude"] = "lat"
    if "longitude" in ds.coords:
        ren["longitude"] = "lon"
    if ren:
        ds = ds.rename(ren)

    # basic presence check
    for c in ("time", "lat", "lon"):
        if c not in ds.coords and c not in ds.variables:
            raise KeyError(f"Missing coordinate '{c}'")

    # ensure coords, not data variables
    for c in ("time", "lat", "lon"):
        if c in ds and c not in ds.coords:
            ds = ds.set_coords(c)

    # wrap & sort
    if "lon" in ds.coords:
        lon2 = wrap_lon_180(ds["lon"])
        ds = ds.assign_coords(lon=lon2)
        ds = ds.sortby("lon")
    if "lat" in ds.coords and ds["lat"].ndim == 1:
        lat = ds["lat"].values
        if lat.size >= 2 and (lat[1] - lat[0] < 0):
            ds = ds.sortby("lat")

    # try to decode CF-times if necessary (no hard fail)
    try:
        ds = xr.decode_cf(ds)
    except Exception:
        pass

    return ds


def alias_vars(ds: xr.Dataset) -> xr.Dataset:
    ren: Dict[str, str] = {}
    for v in list(ds.data_vars):
        short = ALIASES.get(v)
        if short and short not in ds.data_vars:
            ren[v] = short
    return ds.rename(ren) if ren else ds


def keep_numeric_vars(ds: xr.Dataset) -> xr.Dataset:
    keep = [k for k in ds.data_vars if np.issubdtype(ds[k].dtype, np.number)]
    return ds[keep] if keep else xr.Dataset(coords=ds.coords)


def open_one_nc(path: Path) -> xr.Dataset:
    """
    Open a single NetCDF file, standardise coords, alias variables,
    drop non-numeric data vars, and return a *loaded* Dataset.

    Tries multiple engines for robustness and ensures the returned
    Dataset is detached from an open file handle.
    """
    last_err: Exception | None = None
    for eng in ("netcdf4", "h5netcdf", None):
        try:
            with xr.open_dataset(path, engine=eng) as ds:
                ds = standardize_coords(ds)
                ds = alias_vars(ds)
                ds = keep_numeric_vars(ds)
                # Load into memory so we can safely close the underlying file.
                return ds.load()
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to open/standardize {path} with any engine: {last_err}")


def merge_month(files: List[Path], out_path: Path, quiet: bool = False) -> None:
    if out_path.exists():
        if quiet:
            print(f"[skip] exists: {out_path}")
        return

    tmp = out_path.with_suffix(out_path.suffix + ".part")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    if not quiet:
        print(f"[merge] {out_path.name}: {len(files)} file(s)")

    merged: xr.Dataset | None = None

    for i, p in enumerate(sorted(files)):
        if not quiet:
            print(f"   - open {p.name}")
        ds = open_one_nc(p)

        # Force canonical dim order when possible
        order = [d for d in ("time", "lat", "lon") if d in ds.dims]
        ds = ds.transpose(*order, ...)

        if merged is None:
            merged = ds
        else:
            merged = xr.merge([merged, ds], compat="override", join="outer")

    if merged is None:
        raise RuntimeError("No datasets merged (empty month bucket).")

    # Ensure everything is materialised before writing
    merged.load()

    # write compressed NetCDF
    encoding = {v: {"zlib": True, "complevel": 1} for v in merged.data_vars}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_netcdf(tmp, encoding=encoding)
    tmp.replace(out_path)
    if not quiet:
        print(f"[ok] wrote {out_path}")

    try:
        merged.close()
    except Exception:
        pass


def main():
    args = parse_args()
    in_paths = [Path(p) for p in glob.glob(args.in_glob, recursive=True)]
    in_paths = [p for p in in_paths if p.is_file() and p.suffix.lower() == ".nc"]

    if not in_paths:
        msg = f"[err] no .nc matched: {args.in_glob}"
        print(msg, file=sys.stderr)
        sys.exit(2)

    # bucket by YYYYMM
    buckets: Dict[str, List[Path]] = {}
    for p in in_paths:
        tok = find_month_token(p)
        if not tok:
            print(f"[warn] could not find YYYYMM in {p.name}; skipping")
            continue
        buckets.setdefault(tok, []).append(p)

    if not buckets:
        print("[err] no monthly groups found in inputs", file=sys.stderr)
        sys.exit(2)

    out_root = Path(args.out_dir)

    for yyyymm, files in sorted(buckets.items()):
        year = yyyymm[:4]
        month = yyyymm[4:]
        out_path = out_root / year / month / f"era5_single_{yyyymm}_{args.suffix}.nc"
        if out_path.exists() and not args.overwrite:
            if not args.quiet:
                print(f"[skip] exists: {out_path}")
            continue
        merge_month(files, out_path, quiet=args.quiet)

    print("[done] merge complete.")


if __name__ == "__main__":
    main()