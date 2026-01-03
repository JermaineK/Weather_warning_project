#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_features_grid.py
Flatten ERA5 NetCDFs into a tidy grid feature table:
time (UTC, tz-naive), lat, lon, and chosen variables (+ optional derived wspd, zeta, div, S, agree).

Fixes:
  • Accepts --out and --out-features (aliases).
  • Opt-in keeping of u/v via --export-uv.
  • --force-keep to pin variables that must be retained if present.
"""

from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- CLI ----------------

def parse_args():
    ap = argparse.ArgumentParser(description="Flatten ERA5 NetCDFs to grid features (time, lat, lon, vars).")
    # Inputs
    ap.add_argument("--nc-glob", default=None, help="Glob of NetCDFs (e.g. data/**/*.nc)")
    ap.add_argument("--nc", action="append", default=None, help="Add one or more explicit .nc paths (repeatable)")

    # Output (aliases)
    out_group = ap.add_mutually_exclusive_group(required=True)
    out_group.add_argument("--out", dest="out_features", help="Output CSV(.gz) or Parquet path")
    out_group.add_argument("--out-features", dest="out_features", help="Output CSV(.gz) or Parquet path (alias)")

    # Variable names
    ap.add_argument("--uvar", default=None, help="u-component (e.g., u10)")
    ap.add_argument("--vvar", default=None, help="v-component (e.g., v10)")
    ap.add_argument("--mslvar", default=None, help="mean sea level pressure var (e.g., msl)")
    ap.add_argument("--t2mvar", default=None, help="2m temperature var (e.g., t2m)")

    ap.add_argument("--require-vars", nargs="*", default=None,
                    help="If set, skip any NetCDF that does not contain ALL of these variables.")
    ap.add_argument("--force-keep", nargs="*", default=None,
                    help="Optional list of variables to keep if present (e.g., cape cin blh).")

    # Coord names
    ap.add_argument("--time-name", default=None)
    ap.add_argument("--lat-name",  default=None)
    ap.add_argument("--lon-name",  default=None)

    # Domain / thinning
    ap.add_argument("--normalize-lon", choices=["none","-180..180","0..360"], default="none")
    ap.add_argument("--area", default=None, help="latN,lonW,latS,lonE  (match lon range to normalize-lon)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--delta-hours", type=int, default=1)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end",   default=None)

    # Grid identity & duplicates
    ap.add_argument("--emit-grid-index", action="store_true")
    ap.add_argument("--dedup", choices=["none","time_lat_lon"], default="none")

    # Derived fields
    ap.add_argument("--with-vortdiv", action="store_true",
                    help="Compute vorticity/divergence per grid cell and derived proxies S, agree.")
    ap.add_argument("--export-uv", action="store_true",
                    help="Include raw u and v wind components in the output frame.")
    return ap.parse_args()

# ---------------- coord normalization ----------------

POSSIBLE_TIME_NAMES = ("time", "valid_time", "forecast_reference_time")
POSSIBLE_LAT_NAMES  = ("lat", "latitude", "Latitude", "nav_lat")
POSSIBLE_LON_NAMES  = ("lon", "longitude", "Longitude", "nav_lon")

def _pick_name(cands, present):
    for c in cands:
        if c in present: return c
    return None

def collapse_expver(ds: xr.Dataset) -> xr.Dataset:
    if "expver" not in ds.dims: return ds
    try:
        values = set(np.array(ds.coords["expver"].values).tolist())
        if 1 in values:
            ds1 = ds.sel(expver=1)
            if 5 in values:
                ds5 = ds.sel(expver=5)
                ds = ds1.combine_first(ds5)
            else:
                ds = ds1
        else:
            ds = ds.isel(expver=0)
    except Exception:
        ds = ds.isel(expver=0)
    return ds.squeeze(drop=True)

def normalize_coords(ds: xr.Dataset, time_name=None, lat_name=None, lon_name=None) -> xr.Dataset:
    ds = collapse_expver(ds)
    present = set(ds.dims) | set(ds.coords)
    t_in  = time_name or _pick_name(POSSIBLE_TIME_NAMES, present)
    la_in = lat_name  or _pick_name(POSSIBLE_LAT_NAMES,  present)
    lo_in = lon_name  or _pick_name(POSSIBLE_LON_NAMES,  present)
    if not all([t_in, la_in, lo_in]):
        raise ValueError(f"Missing coords: time={t_in}, lat={la_in}, lon={lo_in}")
    ren = {}
    if t_in  != "time": ren[t_in]  = "time"
    if la_in != "lat":  ren[la_in] = "lat"
    if lo_in != "lon":  ren[lo_in] = "lon"
    if ren: ds = ds.rename(ren)
    for c in ("time","lat","lon"):
        if c in ds and c not in ds.coords:
            ds = ds.set_coords(c)
    return ds

# ---------------- lon reframing / area ----------------

def reframe_lon_vals(lon_vals: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none": return lon_vals
    if mode == "0..360": return (lon_vals % 360 + 360) % 360
    return ((lon_vals + 180) % 360) - 180

def reframe_lon_ds(ds: xr.Dataset, mode: str) -> xr.Dataset:
    if mode == "none": return ds
    lon2 = reframe_lon_vals(ds["lon"].to_numpy(), mode)
    order = np.argsort(lon2)
    ds = ds.assign_coords(lon=("lon", lon2))
    if not np.all(order == np.arange(len(lon2))):
        ds = ds.sortby("lon")
    return ds

def parse_area(aoi: str | None):
    if not aoi: return None
    latN, lonW, latS, lonE = [float(x.strip()) for x in aoi.split(",")]
    return latN, lonW, latS, lonE

def select_aoi_ds(ds: xr.Dataset, aoi):
    if not aoi: return ds
    latN, lonW, latS, lonE = aoi
    ds = ds.sel(lon=slice(lonW, lonE))
    lat_vals = ds["lat"].values
    if lat_vals[0] <= lat_vals[-1]:
        ds = ds.sel(lat=slice(latS, latN))
    else:
        ds = ds.sel(lat=slice(latN, latS))
    return ds

# ---------------- thinning utilities ----------------

def keep_delta_hours(df: pd.DataFrame, delta: int) -> pd.DataFrame:
    if delta is None or delta <= 1: return df
    t = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    return df.loc[(t.dt.hour.to_numpy() % int(delta)) == 0].copy()

def apply_area_df(df: pd.DataFrame, aoi):
    if not aoi: return df
    latN, lonW, latS, lonE = aoi
    return df.loc[
        (df["lat"] <= latN) & (df["lat"] >= latS) &
        (df["lon"] >= lonW) & (df["lon"] <= lonE)
    ].copy()

def stride_df(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    if stride <= 1 or df.empty: return df
    df = df.sort_values(["time","lat","lon"], kind="mergesort").reset_index(drop=True)
    def _per_time(g):
        lats = np.sort(g["lat"].unique())
        lons = np.sort(g["lon"].unique())
        lat_map = {v:i for i,v in enumerate(lats)}
        lon_map = {v:i for i,v in enumerate(lons)}
        gi = g.copy()
        gi["_ilat"] = g["lat"].map(lat_map).to_numpy()
        gi["_ilon"] = g["lon"].map(lon_map).to_numpy()
        keep = (gi["_ilat"] % stride == 0) & (gi["_ilon"] % stride == 0)
        return gi.loc[keep].drop(columns=["_ilat","_ilon"])
    return df.groupby(pd.to_datetime(df["time"]).dt.floor("h"), sort=False, group_keys=False).apply(_per_time)

# ---------------- vorticity/divergence helpers ----------------

def compute_zeta_div(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    """Latitude-aware vorticity (zeta) and divergence (div) in s^-1."""
    m_per_deg_y = 110_540.0
    m_per_deg_x_row = 111_320.0 * np.cos(np.deg2rad(lat))
    dlat_deg = np.gradient(lat)
    dlon_deg = np.gradient(lon)
    y_m = np.cumsum(np.r_[0.0, m_per_deg_y * dlat_deg[1:]])
    x_m_nominal = np.cumsum(np.r_[0.0, (m_per_deg_x_row[0] * dlon_deg[1:])])
    dU_dy, dU_dx = np.gradient(u, y_m, x_m_nominal, edge_order=1)
    dV_dy, dV_dx = np.gradient(v, y_m, x_m_nominal, edge_order=1)
    scale2d = (m_per_deg_x_row / m_per_deg_x_row[0])[:, None]
    dU_dx = dU_dx * scale2d
    dV_dx = dV_dx * scale2d
    zeta = dV_dx - dU_dy
    div  = dU_dx + dV_dy
    return zeta, div

# ---------------- dataframe conversion ----------------

def to_frame(ds: xr.Dataset, keep_vars: list[str]) -> pd.DataFrame:
    keep_vars = [v for v in keep_vars if v in ds.data_vars]
    if not keep_vars:
        raise ValueError("No requested variables found in dataset.")
    sub = ds[keep_vars]
    df = sub.to_dataframe().reset_index()
    for c in ("time","lat","lon"):
        if c not in df.columns:
            if c in df.index.names:
                df = df.reset_index(c)
            else:
                raise KeyError(f"Missing '{c}' column after to_dataframe(); columns={list(df.columns)}")
    return df.dropna(subset=["time","lat","lon"])

# ---------------- main ----------------

def main():
    args = parse_args()
    files = []
    if args.nc_glob: files += glob.glob(args.nc_glob, recursive=True)
    if args.nc: files += [p for p in args.nc if p]
    files = sorted({str(f) for f in files if Path(f).exists()})
    if not files:
        print("[err] No NetCDF files found.", file=sys.stderr)
        sys.exit(2)

    requested_vars = []
    if args.uvar:   requested_vars.append(args.uvar)
    if args.vvar:   requested_vars.append(args.vvar)
    if args.mslvar: requested_vars.append(args.mslvar)
    if args.t2mvar: requested_vars.append(args.t2mvar)
    force_keep = set(args.force_keep or [])
    required = set(args.require_vars or [])

    area_box = parse_area(args.area)
    out_path = Path(args.out_features)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dfs = []
    ok_files = skipped_require = 0

    for i, f in enumerate(files, 1):
        try:
            ds = xr.open_dataset(f, engine="netcdf4")
        except Exception:
            ds = xr.open_dataset(f)
        ds = normalize_coords(ds, args.time_name, args.lat_name, args.lon_name)
        ds = reframe_lon_ds(ds, args.normalize_lon)
        ds = select_aoi_ds(ds, area_box)

        present_vars = set(ds.data_vars)
        if required and not required.issubset(present_vars):
            print(f"[skip {i}/{len(files)}] {Path(f).name}: missing {sorted(required - present_vars)}")
            skipped_require += 1
            ds.close()
            continue

        keep_vars = list(requested_vars)

        # Winds and derived fields
        if args.uvar and args.vvar and args.uvar in ds.data_vars and args.vvar in ds.data_vars:
            U = ds[args.uvar].astype("float32")
            V = ds[args.vvar].astype("float32")
            if args.export_uv:
                keep_vars += [args.uvar, args.vvar]
            wspd = np.sqrt(U**2 + V**2)
            ds = ds.assign(wspd=wspd.astype("float32"))
            keep_vars.append("wspd")

            if args.with_vortdiv:
                lat_vals = ds["lat"].values
                lon_vals = ds["lon"].values
                times = ds["time"].values
                z_stack, d_stack = [], []
                for t in times:
                    u2 = U.sel(time=t).values
                    v2 = V.sel(time=t).values
                    zeta, div = compute_zeta_div(u2, v2, lat_vals, lon_vals)
                    z_stack.append(zeta.astype(np.float32))
                    d_stack.append(div.astype(np.float32))
                zeta_da = xr.DataArray(np.stack(z_stack, axis=0),
                                       coords={"time": ds["time"], "lat": ds["lat"], "lon": ds["lon"]},
                                       dims=("time","lat","lon"), name="zeta")
                div_da  = xr.DataArray(np.stack(d_stack, axis=0),
                                       coords={"time": ds["time"], "lat": ds["lat"], "lon": ds["lon"]},
                                       dims=("time","lat","lon"), name="div")
                ds = ds.assign(zeta=zeta_da, div=div_da)
                keep_vars += ["zeta","div"]
                S = np.sqrt(zeta_da**2 + div_da**2)
                ds = ds.assign(S=S.astype("float32"))
                keep_vars.append("S")
                agree = (np.abs(zeta_da) > np.abs(div_da)).astype("float32")
                ds = ds.assign(agree=agree)
                keep_vars.append("agree")

        # msl sanity (hectopascals if looks like Pascals)
        if args.mslvar and args.mslvar in ds.data_vars:
            msl_da = ds[args.mslvar].astype("float32")
            try:
                med = float(np.nanmedian(msl_da.values))
                if med > 2000.0: msl_da = msl_da / 100.0
            except Exception:
                pass
            ds = ds.assign(msl=msl_da)
            keep_vars.append("msl")

        # Force-keep extras if present (e.g., CAPE/CIN/etc.)
        for v in list(force_keep):
            if v in ds.data_vars:
                keep_vars.append(v)

        # Default fallback: keep all numeric vars if user didn’t request any
        if not keep_vars:
            keep_vars = [k for k in ds.data_vars if np.issubdtype(ds[k].dtype, np.number)]

        try:
            df = to_frame(ds, keep_vars)
        except Exception as e:
            print(f"[skip {i}] {Path(f).name}: to_frame error {e}")
            ds.close(); continue

        df = keep_delta_hours(df, args.delta_hours)
        df = apply_area_df(df, area_box)
        if args.normalize_lon != "none":
            df["lon"] = reframe_lon_vals(pd.to_numeric(df["lon"], errors="coerce").to_numpy(), args.normalize_lon)
        if args.stride > 1:
            df = stride_df(df, args.stride)
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=["time","lat","lon"]).reset_index(drop=True)

        if not df.empty:
            dfs.append(df); ok_files += 1
            kept = sorted(set(df.columns) - {"time","lat","lon"})
            print(f"[{i}/{len(files)}] {Path(f).name}: rows={len(df):,}  vars={kept}")
        ds.close()

    if not dfs:
        print(f"[err] No rows produced. Files read={ok_files}, skipped_missing_required={skipped_require}", file=sys.stderr)
        sys.exit(1)

    out_df = pd.concat(dfs, ignore_index=True)
    if args.dedup == "time_lat_lon":
        out_df = out_df.drop_duplicates(subset=["time","lat","lon"], keep="first", ignore_index=True)

    if args.emit_grid_index and not out_df.empty:
        lats = np.sort(out_df["lat"].unique())
        lons = np.sort(out_df["lon"].unique())
        out_df["ilat"] = out_df["lat"].map({v:i for i,v in enumerate(lats)}).astype("int32")
        out_df["ilon"] = out_df["lon"].map({v:i for i,v in enumerate(lons)}).astype("int32")

    out_df.sort_values(["time","lat","lon"], inplace=True, ignore_index=True)

    low = out_path.name.lower()
    if low.endswith((".parquet",".parq",".pq")):
        out_df.to_parquet(out_path, index=False)
    else:
        comp = "gzip" if low.endswith(".csv.gz") or out_path.suffix.lower()==".gz" else "infer"
        out_df.to_csv(out_path, index=False, compression=comp, date_format="%Y-%m-%d %H:%M:%S")

    H = out_df["lat"].nunique(); W = out_df["lon"].nunique(); T = out_df["time"].nunique()
    print(f"[ok] wrote {len(out_df):,} rows -> {out_path}  (H={H} x W={W} x T={T})")
    if args.emit_grid_index:
        print("  (ilat/ilon present -> stable grid IDs across hours)")

if __name__ == "__main__":
    main()