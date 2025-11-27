#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integrate_era5_thermo.py

Integrate ERA5 single-level fields (u10, v10, msl, t2m) back into a large
feature table (e.g. grid_labelled_FMA_gka.parquet), month-by-month, in a
memory-conscious way.

Key points
----------
* Input features: CSV/CSV.GZ/Parquet, with columns at least: time, lat, lon
* ERA5 thermo: one or more NetCDFs (e.g. era5_single_YYYYMM_oper.nc)
* Matching is on (time, lat, lon) using ERA5’s own grid coordinates
* No fragile integer index tricks; a preserved __idx column is used when
  writing merged thermo values back into the big features DataFrame.

CLI (example)
-------------
python integrate_era5_thermo.py \
    --features data/grid_labelled_FMA_gka.parquet \
    --thermo-glob "data_era5/extracted/**/era5_single_*_*.nc" \
    --out data/grid_labelled_FMA_gka_realthermo.parquet \
    --normalize-lon "-180..180" \
    --area "-5,125,-35,175" \
    --nearest --nearest-maxdeg 0.4
"""

from __future__ import annotations
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

pd.options.mode.copy_on_write = True

# ---------- ERA5 alias and coord helpers (borrowed from build_features_grid.py) ----------

ALIASES = {
    "u10": ["u10", "10m_u_component_of_wind", "U10M", "u_10m"],
    "v10": ["v10", "10m_v_component_of_wind", "V10M", "v_10m"],
    "msl": ["msl", "mean_sea_level_pressure", "MSL", "prmsl"],
    "t2m": ["t2m", "2m_temperature", "T2M", "t_2m"],
}

POSSIBLE_TIME_NAMES = ("time", "valid_time", "forecast_reference_time")
POSSIBLE_LAT_NAMES  = ("lat", "latitude", "Latitude", "nav_lat")
POSSIBLE_LON_NAMES  = ("lon", "longitude", "Longitude", "nav_lon")


def _pick_coord_name(cands, present):
    for c in cands:
        if c in present:
            return c
    return None


def _resolve_alias(name: str | None, present: set[str], key: str) -> str | None:
    """
    Resolve a variable name inside an xarray Dataset, using an explicit name
    if provided, otherwise an alias list for that key.
    """
    if name and name in present:
        return name
    if key in ALIASES:
        for cand in ALIASES[key]:
            if cand in present:
                return cand
    # if user supplied one of the alias strings explicitly, map it to the
    # first present candidate
    if name and key in ALIASES and name in ALIASES[key]:
        for cand in ALIASES[key]:
            if cand in present:
                return cand
    return None


def collapse_expver(ds: xr.Dataset) -> xr.Dataset:
    """Handle expver dimension in ERA5 (1/5 merge) the same way as build_features_grid."""
    if "expver" not in ds.dims:
        return ds
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


def normalize_coords(ds: xr.Dataset,
                     time_name: str | None = None,
                     lat_name: str | None = None,
                     lon_name: str | None = None) -> xr.Dataset:
    ds = collapse_expver(ds)
    present = set(ds.dims) | set(ds.coords)
    t_in  = time_name or _pick_coord_name(POSSIBLE_TIME_NAMES, present)
    la_in = lat_name  or _pick_coord_name(POSSIBLE_LAT_NAMES,  present)
    lo_in = lon_name  or _pick_coord_name(POSSIBLE_LON_NAMES,  present)
    if not all([t_in, la_in, lo_in]):
        raise ValueError(f"Missing coords: time={t_in}, lat={la_in}, lon={lo_in}")

    ren = {}
    if t_in  != "time": ren[t_in]  = "time"
    if la_in != "lat":  ren[la_in] = "lat"
    if lo_in != "lon":  ren[lo_in] = "lon"
    if ren:
        ds = ds.rename(ren)
    for c in ("time", "lat", "lon"):
        if c in ds and c not in ds.coords:
            ds = ds.set_coords(c)
    return ds


def _canon_norm(mode: str | None) -> str:
    mode = (mode or "none").strip()
    if mode in ("-180..180", "0..360", "none"):
        return mode
    if mode.replace(" ", "") == "-180..180":
        return "-180..180"
    if mode.replace(" ", "") == "0..360":
        return "0..360"
    return "none"


def reframe_lon_vals(lon_vals: np.ndarray, mode: str) -> np.ndarray:
    mode = _canon_norm(mode)
    if mode == "none":
        return lon_vals
    if mode == "0..360":
        return (lon_vals % 360 + 360) % 360
    # default -180..180
    return ((lon_vals + 180) % 360) - 180


def reframe_lon_ds(ds: xr.Dataset, mode: str) -> xr.Dataset:
    mode = _canon_norm(mode)
    if mode == "none":
        return ds
    lon2 = reframe_lon_vals(ds["lon"].to_numpy(), mode)
    order = np.argsort(lon2)
    ds = ds.assign_coords(lon=("lon", lon2))
    if not np.all(order == np.arange(len(lon2))):
        ds = ds.sortby("lon")
    return ds


def parse_area(aoi: str | None):
    if not aoi:
        return None
    latN, lonW, latS, lonE = [float(x.strip()) for x in aoi.split(",")]
    return latN, lonW, latS, lonE


def select_aoi_ds(ds: xr.Dataset, aoi):
    if not aoi:
        return ds
    latN, lonW, latS, lonE = aoi
    ds = ds.sel(lon=slice(lonW, lonE))
    lat_vals = ds["lat"].values
    if lat_vals[0] <= lat_vals[-1]:
        ds = ds.sel(lat=slice(latS, latN))
    else:
        ds = ds.sel(lat=slice(latN, latS))
    return ds


# ---------- generic I/O helpers ----------

def load_any_table(path: str) -> pd.DataFrame:
    """Load features file (CSV/CSV.GZ/Parquet)."""
    p = str(path).lower()
    if p.endswith((".parquet", ".parq", ".pq")):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(
            path,
            compression="infer",
            low_memory=False,
            encoding_errors="replace",
            on_bad_lines="skip",
            parse_dates=["time"],
        )
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    return df


def write_any_table(path: str, df: pd.DataFrame) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    low = p.name.lower()
    if low.endswith((".parquet", ".parq", ".pq")):
        df.to_parquet(p, index=False)
    else:
        comp = "gzip" if (low.endswith(".csv.gz") or p.suffix.lower() == ".gz") else "infer"
        df.to_csv(p, index=False, compression=comp, date_format="%Y-%m-%d %H:%M:%S")


# ---------- thermo flattening ----------

def flatten_era5_single(path: str,
                        normalize_lon: str,
                        area_box,
                        engine: str | None = None) -> pd.DataFrame:
    """
    Open a single ERA5 single-level file, extract u10,v10,msl,t2m (if present),
    normalize lon + AOI, and flatten to a tidy DataFrame with time,lat,lon,...
    """
    print(f"[thermo] open {Path(path).name}", flush=True)
    # engine hint + fallback
    engines = [engine] if engine else []
    engines += ["netcdf4", "h5netcdf", "scipy"]
    ds = None
    tried = []
    for eng in engines:
        if eng is None:
            continue
        try:
            ds = xr.open_dataset(path, engine=eng)
            break
        except Exception as e:
            tried.append((eng, str(e)[:80]))
    if ds is None:
        # last resort: let xarray guess
        ds = xr.open_dataset(path)

    ds = normalize_coords(ds)
    ds = reframe_lon_ds(ds, normalize_lon)
    ds = select_aoi_ds(ds, area_box)

    present_vars = set(ds.data_vars)
    u_name = _resolve_alias(None, present_vars, "u10")
    v_name = _resolve_alias(None, present_vars, "v10")
    msl_name = _resolve_alias(None, present_vars, "msl")
    t2m_name = _resolve_alias(None, present_vars, "t2m")

    keep = []
    if u_name and v_name:
        keep += [u_name, v_name]
    if msl_name:
        keep.append(msl_name)
    if t2m_name:
        keep.append(t2m_name)

    if not keep:
        print(f"[thermo] {Path(path).name}: no core vars (u10/v10/msl/t2m) found; skipping.")
        ds.close()
        return pd.DataFrame(columns=["time","lat","lon"])

    sub = ds[keep]
    # downcast to float32 for IO savings
    for v in list(sub.data_vars):
        if np.issubdtype(sub[v].dtype, np.floating):
            sub[v] = sub[v].astype("float32")

    df = sub.to_dataframe().reset_index()
    # ensure time/lat/lon are columns, drop any all-NaN rows
    for c in ("time","lat","lon"):
        if c not in df.columns and c in df.index.names:
            df = df.reset_index(c)
    df = df.dropna(subset=["time","lat","lon"])
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    df["lat"]  = pd.to_numeric(df["lat"], errors="coerce").astype("float32")
    df["lon"]  = pd.to_numeric(df["lon"], errors="coerce").astype("float32")

    # unify var names to canonical u10,v10,msl,t2m
    ren = {}
    if u_name and u_name != "u10": ren[u_name] = "u10"
    if v_name and v_name != "v10": ren[v_name] = "v10"
    if msl_name and msl_name != "msl": ren[msl_name] = "msl"
    if t2m_name and t2m_name != "t2m": ren[t2m_name] = "t2m"
    if ren:
        df = df.rename(columns=ren)

    # auto-convert MSL Pa→hPa if needed
    if "msl" in df.columns:
        try:
            med = float(np.nanmedian(df["msl"].to_numpy()))
            if med > 2000.0:  # pretty safe Pa vs hPa separator
                df["msl"] = df["msl"] / 100.0
        except Exception:
            pass

    keep_cols = ["time","lat","lon"]
    for c in ("u10","v10","msl","t2m"):
        if c in df.columns:
            keep_cols.append(c)
    df = df[keep_cols].sort_values(["time","lat","lon"], kind="mergesort").reset_index(drop=True)

    print(f"[thermo] {Path(path).name}: rows={len(df):,} vars={keep_cols[3:]}", flush=True)
    ds.close()
    return df


# ---------- CLI + main ----------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Integrate ERA5 single-level thermo fields back into a feature table."
    )
    ap.add_argument("--features", required=True,
                    help="Input feature table (CSV/CSV.GZ/Parquet) with time,lat,lon.")
    ap.add_argument("--thermo-glob", default=None,
                    help="Glob for ERA5 single-level NetCDFs (e.g. data_era5/extracted/**/era5_single_*_*.nc)")
    ap.add_argument("--nc-glob", default=None,
                    help="Alias for --thermo-glob (for older YAMLs).")
    ap.add_argument("--out", required=True,
                    help="Output features with thermo columns added/overwritten.")
    ap.add_argument("--engine", default=None,
                    help="Optional xarray engine hint: netcdf4, h5netcdf, scipy, cfgrib")
    ap.add_argument("--normalize-lon", default="-180..180",
                    help="Lon mode: ' -180..180', '-180..180', '0..360', 'none'")
    ap.add_argument("--area", default=None,
                    help='Optional AOI "latN,lonW,latS,lonE" applied when reading ERA5.')
    # kept for compatibility; we currently do exact (time,lat,lon) join
    ap.add_argument("--nearest", action="store_true",
                    help="(Currently a no-op: exact (time,lat,lon) merge is used.)")
    ap.add_argument("--nearest-maxdeg", type=float, default=0.4,
                    help="Unused placeholder for future nearest-neighbour matching.")
    return ap.parse_args()


def main():
    args = parse_args()
    feat_path = Path(args.features)
    out_path = Path(args.out)

    thermo_glob = args.thermo_glob or args.nc_glob
    if not thermo_glob:
        raise SystemExit("Must supply --thermo-glob or --nc-glob")

    nc_files = sorted(glob.glob(thermo_glob, recursive=True))
    if not nc_files:
        raise SystemExit(f"No ERA5 thermo files found for pattern: {thermo_glob}")

    area_box = parse_area(args.area)
    lon_mode = _canon_norm(args.normalize_lon)

    print(f"[integrate] features: {feat_path}")
    print(f"[integrate] thermo glob: {thermo_glob} → {len(nc_files)} files")
    print(f"[integrate] lon mode: {lon_mode}  AOI: {area_box}", flush=True)
    if args.nearest:
        print("[integrate] --nearest requested; using exact (time,lat,lon) join for now.", flush=True)

    # Load feature table once
    feat = load_any_table(str(feat_path))
    if not {"time","lat","lon"}.issubset(feat.columns):
        raise SystemExit("Feature table must include columns: time, lat, lon")

    # normalize coords same way as thermo
    feat["time"] = pd.to_datetime(feat["time"], utc=True, errors="coerce").dt.tz_localize(None)
    feat["lat"]  = pd.to_numeric(feat["lat"], errors="coerce").astype("float32")
    feat["lon"]  = reframe_lon_vals(pd.to_numeric(feat["lon"], errors="coerce").to_numpy(), lon_mode).astype("float32")
    feat = feat.dropna(subset=["time","lat","lon"]).reset_index(drop=True)

    # stable ordering, preserve index for later
    feat.sort_values(["time","lat","lon"], kind="mergesort", inplace=True, ignore_index=True)

    # We'll add thermo columns if missing
    for c in ("u10","v10","msl","t2m"):
        if c not in feat.columns:
            feat[c] = np.nan

    # Process each ERA5 file month-by-month
    for i, nc in enumerate(nc_files, start=1):
        thermo_df = flatten_era5_single(nc, lon_mode, area_box, engine=args.engine)
        if thermo_df.empty:
            print(f"[integrate] {Path(nc).name}: no thermo rows after AOI; skipping", flush=True)
            continue

        # restrict features to time range of this thermo chunk
        tmin = thermo_df["time"].min()
        tmax = thermo_df["time"].max()
        mask = (feat["time"] >= tmin) & (feat["time"] <= tmax)
        if not mask.any():
            print(f"[integrate] {Path(nc).name}: no feature rows in [{tmin}, {tmax}]; skipping", flush=True)
            continue

        feat_chunk = feat.loc[mask, ["time","lat","lon"]].copy()
        feat_chunk["__idx"] = feat_chunk.index  # preserve original indices into main feat

        # Merge thermo onto this time-slice
        merged = feat_chunk.merge(
            thermo_df,
            on=["time","lat","lon"],
            how="left",
            suffixes=("","_thermo")
        )

        # Write back into feat, using preserved __idx
        idx = merged["__idx"].to_numpy()
        for col in ("u10","v10","msl","t2m"):
            if col in merged.columns:
                vals = pd.to_numeric(merged[col], errors="coerce").to_numpy()
                feat.loc[idx, col] = vals

        print(f"[integrate] {i}/{len(nc_files)} {Path(nc).name}: "
              f"feat_rows_in_range={mask.sum():,} thermo_rows={len(thermo_df):,}", flush=True)

    # done
    write_any_table(str(out_path), feat)
    print(f"[integrate] wrote {out_path} rows={len(feat):,}", flush=True)


if __name__ == "__main__":
    main()
