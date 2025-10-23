#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integrate ERA5 single-level fields into the labelled grid and build
robust warm–moist / surface-flux thermo proxies.

Features:
  • Reads labelled from CSV(.gz) or Parquet; writes CSV(.gz) or Parquet (by --out extension)
  • open_mfdataset by coords (tolerant engines)
  • Time normalized to tz-naive UTC
  • Longitude wrap for BOTH labelled & ERA5 to [-180, 180)
  • Accumulated→hourly conversion for tp/slhf/sshf
  • Optional nearest-neighbour snapping with max-degree cap
  • Proxy construction only when inputs present

Example:
  python integrate_era5_thermo.py ^
    --labelled data/grid_labelled_FMA_gka.csv.gz ^
    --nc-glob "data_era5/extracted/**/data_stream-oper_stepType-*.nc" ^
    --out data/grid_labelled_FMA_gka_realthermo.parquet ^
    --nearest --nearest-maxdeg 0.4
"""

import argparse
import glob
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr


# ------------------------- small I/O helpers -------------------------

def read_any(path: str, parse_time: bool = True) -> pd.DataFrame:
    low = path.lower()
    if low.endswith((".parquet", ".parq", ".pq")):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, compression="infer", low_memory=False)
    if parse_time and "time" in df.columns:
        df["time"] = _tz_to_naive_utc(df["time"])
    return df


def write_any(path: str, df: pd.DataFrame) -> None:
    low = path.lower()
    if low.endswith((".parquet", ".parq", ".pq")):
        df.to_parquet(path, index=False)
    else:
        comp = "gzip" if low.endswith(".gz") else "infer"
        df.to_csv(path, index=False, compression=comp)


# ------------------------- utilities -------------------------

def _tz_to_naive_utc(s: pd.Series) -> pd.Series:
    """Coerce a datetime-like Series to tz-naive UTC."""
    s = pd.to_datetime(s, utc=True, errors="coerce")
    return s.dt.tz_convert("UTC").dt.tz_localize(None)


def _wrap_lon_180(x: pd.Series) -> pd.Series:
    """Wrap longitudes to [-180, 180)."""
    xn = pd.to_numeric(x, errors="coerce")
    return ((xn + 180.0) % 360.0) - 180.0


def _mad(series: pd.Series) -> float:
    """Mean absolute deviation from median (robust to NaNs)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0
    med = float(np.median(s))
    return float(np.mean(np.abs(s - med)))


def _try_open_mfdataset(files: List[str]) -> xr.Dataset:
    """Open multiple NetCDF files with a tolerant engine strategy."""
    try:
        return xr.open_mfdataset(files, combine="by_coords", parallel=False)
    except Exception:
        try:
            return xr.open_mfdataset(files, combine="by_coords", parallel=False, engine="netcdf4")
        except Exception:
            return xr.open_mfdataset(files, combine="by_coords", parallel=False, engine="h5netcdf")


def _standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    """Ensure time/lat/lon exist with canonical names and lon in [-180, 180)."""
    if "time" not in ds.coords and "time" not in ds.variables:
        if "valid_time" in ds.coords or "valid_time" in ds.variables:
            ds = ds.rename({"valid_time": "time"})
        else:
            raise KeyError("No 'time' or 'valid_time' coordinate found in ERA5 dataset.")

    if "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.coords:
        ds = ds.rename({"longitude": "lon"})

    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise KeyError("No 'lat'/'lon' coordinates found in ERA5 dataset.")

    # Wrap longitudes to [-180, 180)
    lon = ds["lon"]
    if float(lon.max()) > 180.0:
        new_lon = ((lon + 180.0) % 360.0) - 180.0
        ds = ds.assign_coords(lon=new_lon).sortby("lon")

    return ds


def _era5_to_frame(ds: xr.Dataset, keep_vars: List[str]) -> pd.DataFrame:
    """Flatten ERA5 dataset to DataFrame with columns: time, lat, lon, <vars>."""
    cols_present = [v for v in keep_vars if v in ds.data_vars]
    if not cols_present:
        raise ValueError("None of the requested variables are present in the ERA5 dataset.")

    sub = ds[cols_present]
    df = sub.to_dataframe().reset_index()

    if "time" not in df.columns:
        raise KeyError("Flattened ERA5 dataframe has no 'time' column.")
    df["time"] = _tz_to_naive_utc(df["time"])

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = _wrap_lon_180(pd.to_numeric(df["lon"], errors="coerce"))
    df = df.dropna(subset=["lat", "lon", "time"]).reset_index(drop=True)
    return df


def _convert_accum_to_hourly(df: pd.DataFrame, var: str) -> pd.DataFrame:
    """Convert an accumulated variable to hourly totals (per lat,lon diffs)."""
    if var not in df.columns:
        return df
    df = df.sort_values(["lat", "lon", "time"], kind="mergesort").reset_index(drop=True)
    d = df.groupby(["lat", "lon"], sort=False)[var].diff()
    d = d.where(d >= 0, np.nan).fillna(0.0).astype(float)
    df[f"{var}_hourly"] = d.to_numpy()
    return df


def _nearest_snap(
    lat_src: np.ndarray, lon_src: np.ndarray,
    lat_tgt: np.ndarray, lon_tgt: np.ndarray,
    max_deg: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each target (lat_tgt, lon_tgt) pick nearest source (regular grid assumption).
    Returns (lat_nn, lon_nn, dist_deg).
    """
    u_lat = np.unique(lat_src)  # sorted ascending
    u_lon = np.unique(lon_src)  # sorted ascending

    def nearest(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(grid, values, side="left")
        idx = np.clip(idx, 1, len(grid) - 1)
        left = grid[idx - 1]
        right = grid[idx]
        choose_right = (right - values) < (values - left)
        return np.where(choose_right, right, left)

    lat_nn = nearest(lat_tgt, u_lat)
    lon_nn = nearest(lon_tgt, u_lon)
    dist = np.sqrt((lat_nn - lat_tgt) ** 2 + (lon_nn - lon_tgt) ** 2)

    mask = dist <= max_deg
    lat_nn = np.where(mask, lat_nn, np.nan)
    lon_nn = np.where(mask, lon_nn, np.nan)
    dist = np.where(mask, dist, np.nan)
    return lat_nn, lon_nn, dist


def add_light_thermo_proxies(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add robust thermo proxies; compute each term only if inputs exist & non-all-NaN.
    Returns (df, added_columns).
    """
    added: List[str] = []

    # Normalised 2m temperature
    if "t2m" in df:
        t2m = pd.to_numeric(df["t2m"], errors="coerce")
        t2m_nd = (t2m - t2m.median()) / (_mad(t2m) + 1e-6)
        df["t2m_nd"] = t2m_nd.astype("float32")
        added.append("t2m_nd")

    # Dewpoint spread
    if {"t2m", "d2m"}.issubset(df.columns):
        spread = pd.to_numeric(df["t2m"], errors="coerce") - pd.to_numeric(df["d2m"], errors="coerce")
        df["t2m_d2m_spread"] = spread.astype("float32")
        added.append("t2m_d2m_spread")

    # TCWV normalised
    if "tcwv" in df:
        tcwv = pd.to_numeric(df["tcwv"], errors="coerce")
        tcwv_nd = (tcwv - tcwv.median()) / (_mad(tcwv) + 1e-6)
        df["tcwv_nd"] = tcwv_nd.astype("float32")
        added.append("tcwv_nd")

    # MSL pressure tendency (1h diff per cell)
    if "msl" in df and {"lat", "lon", "time"}.issubset(df.columns):
        df.sort_values(["lat", "lon", "time"], kind="mergesort", inplace=True, ignore_index=True)
        msl = pd.to_numeric(df["msl"], errors="coerce")
        df["msl_d1h_real"] = msl.groupby([df["lat"], df["lon"]], sort=False).diff().astype("float32")
        added.append("msl_d1h_real")

    # Surface fluxes (if converted to hourly, use *_hourly)
    for raw, out in [("sshf_hourly", "sshf_h"), ("slhf_hourly", "slhf_h")]:
        if raw in df:
            df[out] = pd.to_numeric(df[raw], errors="coerce").astype("float32")
            added.append(out)

    # Warm–moist composite (only if we have at least one term)
    parts = [c for c in ("t2m_nd", "tcwv_nd") if c in df]
    if parts:
        comp = sum(pd.to_numeric(df[c], errors="coerce") for c in parts)
        if "msl" in df:
            msl = pd.to_numeric(df["msl"], errors="coerce")
            msl_nd = (msl - msl.median()) / (_mad(msl) + 1e-6)
            comp = comp - 0.5 * msl_nd
        df["thermo_warm_moist_proxy_real"] = comp.astype("float32")
        added.append("thermo_warm_moist_proxy_real")

    # Simple buoyancy forcing from fluxes
    if {"sshf_h", "slhf_h"}.issubset(df.columns):
        df["thermo_buoyancy_flux"] = (df["sshf_h"] + 0.7 * df["slhf_h"]).astype("float32")
        added.append("thermo_buoyancy_flux")

    return df, added


# ------------------------- main -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ERA5 → real thermo integration")
    p.add_argument("--labelled", required=True, help="Labelled CSV(.gz) or Parquet")
    p.add_argument("--nc-glob", required=True, help="Glob for extracted ERA5 .nc files")
    p.add_argument("--out", required=True, help="Output CSV(.gz) or Parquet (by extension)")
    p.add_argument("--nearest", action="store_true", help="Nearest-neighbour snap to ERA grid")
    p.add_argument("--nearest-maxdeg", type=float, default=0.4, help="Max deg distance for snap")
    return p.parse_args()


def main():
    args = parse_args()
    print("== ERA5 → Real Thermo Integration ==")
    print(f"Labelled : {args.labelled}")
    print(f"NC glob  : {args.nc_glob}")
    print(f"Out      : {args.out}")
    print(f"Opts     : nearest={args.nearest}  nearest_maxdeg={args.nearest_maxdeg}")

    # 1) labelled (read-any) + normalize lon to [-180, 180) to match ERA5
    base = read_any(args.labelled, parse_time=True)
    if not {"time", "lat", "lon"}.issubset(base.columns):
        raise ValueError("Labelled input must contain 'time','lat','lon' columns.")
    base["lat"] = pd.to_numeric(base["lat"], errors="coerce")
    base["lon"] = _wrap_lon_180(base["lon"])
    base = base.dropna(subset=["time", "lat", "lon"]).reset_index(drop=True)
    print(f"[LABELLED] rows={len(base):,} cols={len(base.columns)}")

    # 2) open ERA5
    nc_files = sorted(glob.glob(args.nc_glob, recursive=True))
    if not nc_files:
        raise FileNotFoundError(f"No files matched: {args.nc_glob}")
    print(f"[ERA5] files={len(nc_files)}")

    ds = _try_open_mfdataset(nc_files)
    ds = _standardize_coords(ds)

    # Requested variables (use aliases if GUI names present)
    want_vars = [
        "u10", "v10", "t2m", "d2m", "msl", "sp", "tcwv", "tp", "slhf", "sshf",
        "total_precipitation", "surface_latent_heat_flux", "surface_sensible_heat_flux",
        "total_column_water_vapour",
    ]
    alias: Dict[str, str] = {
        "total_precipitation": "tp",
        "surface_latent_heat_flux": "slhf",
        "surface_sensible_heat_flux": "sshf",
        "total_column_water_vapour": "tcwv",
    }
    present = []
    for w in want_vars:
        if w in ds.data_vars:
            present.append(w)
    for k, v in alias.items():
        if k in ds.data_vars and v not in present:
            ds = ds.rename({k: v})
            present.append(v)

    if not present:
        raise ValueError("No requested ERA5 variables found in dataset.")
    keep_vars = sorted(set(present))
    print(f"[ERA5] keeping variables: {keep_vars}")

    era = _era5_to_frame(ds, keep_vars=keep_vars)
    print(f"[ERA5->DF] rows={len(era):,} cols={len(era.columns)}")

    # 3) accumulations → hourly
    for acc in ("tp", "slhf", "sshf"):
        if acc in era.columns:
            era = _convert_accum_to_hourly(era, acc)

    # 4) nearest snapping (optional)
    if args.nearest:
        lat_nn, lon_nn, dist = _nearest_snap(
            lat_src=era["lat"].to_numpy(),
            lon_src=era["lon"].to_numpy(),
            lat_tgt=base["lat"].to_numpy(),
            lon_tgt=base["lon"].to_numpy(),
            max_deg=args.nearest_maxdeg,
        )
        base["lat_snap"] = lat_nn
        base["lon_snap"] = lon_nn
        ok = ~np.isnan(dist)
        kept_before = len(base)
        base = base[ok].reset_index(drop=True)
        print(f"[SNAP] kept {len(base):,}/{kept_before:,} rows within {args.nearest_maxdeg}°")
        era = era.rename(columns={"lat": "lat_snap", "lon": "lon_snap"})
        merge_keys = ["time", "lat_snap", "lon_snap"]
    else:
        merge_keys = ["time", "lat", "lon"]

    # 5) merge
    keep_cols = ["time", "lat_snap" if args.nearest else "lat", "lon_snap" if args.nearest else "lon"]
    keep_cols += [c for c in era.columns if c not in ("time", "lat", "lon", "lat_snap", "lon_snap")]
    era_small = era[keep_cols]
    merged = base.merge(era_small, on=merge_keys, how="left")
    print(f"[MERGE] rows={len(merged):,} cols={len(merged.columns)}")

    # 6) thermo proxies (tolerant)
    merged, thermo_added = add_light_thermo_proxies(merged)
    if thermo_added:
        print(f"[THERMO] Added: {', '.join(thermo_added)}")
    else:
        print("[THERMO] No thermo proxies added (inputs missing everywhere).")

    # 7) write out (drop snap keys if requested)
    core = ["time", "lat", "lon"]
    if args.nearest:
        merged = merged.drop(columns=["lat_snap", "lon_snap"], errors="ignore")
    out_cols = core + [c for c in merged.columns if c not in core]

    write_any(args.out, merged[out_cols])
    print(f"Wrote {args.out} | rows={len(merged):,} cols={len(out_cols)}")


if __name__ == "__main__":
    main()