#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

# ---------- optional user hook ----------
def _maybe_apply_user_enhancements(df: pd.DataFrame, ds: Optional[xr.Dataset]) -> pd.DataFrame:
    """
    If users provide features_patch.add_all_feature_enhancements, call it.
    Accepts either (df) or (df, ds). Fail-quietly to keep pipeline robust.
    """
    try:
        from features_patch import add_all_feature_enhancements  # optional
        try:
            out = add_all_feature_enhancements(df, ds)  # preferred signature
        except TypeError:
            out = add_all_feature_enhancements(df)       # legacy signature
        return out if out is not None else df
    except Exception:
        return df


# ---------- I/O helpers ----------
def _write_any(path: str, df: pd.DataFrame) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    low = p.name.lower()
    if low.endswith((".parquet", ".parq", ".pq")):
        df.to_parquet(p, index=False)
    else:
        comp = "gzip" if low.endswith(".csv.gz") or p.suffix.lower() == ".gz" else "infer"
        df.to_csv(p, index=False, compression=comp, date_format="%Y-%m-%d %H:%M:%S")


# ---------- utilities ----------
def _expand_inputs(nc: List[str], nc_glob: Optional[str]) -> List[str]:
    """Turn inputs into a flat list of .nc files."""
    files: List[str] = []
    if nc_glob:
        files.extend(glob.glob(nc_glob, recursive=True))
    for p in (nc or []):
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, "*.nc")))
        else:
            files.append(p)
    # dedupe while preserving order
    seen = set(); out = []
    for f in files:
        f = str(Path(f))
        if f.lower().endswith(".nc") and f not in seen:
            out.append(f); seen.add(f)
    return out


def _open_mfdataset_safely(files: List[str]) -> xr.Dataset:
    """Try opening a list of NC files with sensible fallbacks."""
    if not files:
        raise FileNotFoundError("No NetCDF files found (check --nc / --nc-glob).")
    for kwargs in ({}, {"engine":"netcdf4"}, {"engine":"h5netcdf"}):
        try:
            return xr.open_mfdataset(files, combine="by_coords", parallel=False, **kwargs)
        except Exception:
            pass
    # last resort — will raise if it fails
    return xr.open_mfdataset(files, combine="by_coords", parallel=False)


def _normalize_lon_xr(arr: xr.DataArray, mode: str) -> xr.DataArray:
    """Normalize longitude coordinate."""
    v = arr.astype("float64").values
    if mode == "none":
        pass
    elif mode == "-180..180":
        v = ((v + 180.0) % 360.0) - 180.0
    elif mode == "0..360":
        v = v % 360.0
        v[v < 0] += 360.0
    else:
        raise ValueError("normalize-lon must be one of: none, -180..180, 0..360")
    return xr.DataArray(v, dims=arr.dims, coords=arr.coords, attrs=arr.attrs, name=arr.name)


def _standardize_axes(
    ds: xr.Dataset,
    time_name: Optional[str],
    lat_name: Optional[str],
    lon_name: Optional[str],
    normalize_lon: str = "none",
) -> Tuple[xr.Dataset, str, str, str]:
    """Find/rename coord dims to (time, lat, lon); ensure lat↑ lon↑; optional lon normalization."""
    # Time
    tcands = [time_name, "time", "valid_time", "datetime", "forecast_time"]
    tname = next((c for c in tcands if c and c in ds.coords), None)
    if tname is None:
        tname = next((c for c in tcands if c and c in ds.variables), None)
    if tname is None:
        raise ValueError(f"Could not find a time coordinate among {tcands}.")

    # Lat/Lon
    lcands = [lat_name, "latitude", "lat", "Latitude", "nav_lat"]
    latn = next((c for c in lcands if c and c in ds.coords), None)
    if latn is None:
        latn = next((c for c in lcands if c and c in ds.variables), None)
    if latn is None:
        raise ValueError(f"Could not find a latitude coordinate among {lcands}.")

    ocands = [lon_name, "longitude", "lon", "Longitude", "nav_lon"]
    lonn = next((c for c in ocands if c and c in ds.coords), None)
    if lonn is None:
        lonn = next((c for c in ocands if c and c in ds.variables), None)
    if lonn is None:
        raise ValueError(f"Could not find a longitude coordinate among {ocands}.")

    # Rename to canonical
    rename_dict = {}
    if tname != "time": rename_dict[tname] = "time"
    if latn != "lat":   rename_dict[latn]   = "lat"
    if lonn != "lon":   rename_dict[lonn]   = "lon"
    if rename_dict:
        ds = ds.rename(rename_dict)

    # Normalize lon and sort
    ds = ds.assign_coords(lon=_normalize_lon_xr(ds["lon"], normalize_lon))
    ds = ds.sortby(["lat", "lon"])

    return ds, "time", "lat", "lon"


def _safe_percentile(a: np.ndarray, q: float) -> float:
    a = np.asarray(a, dtype="float64")
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 1.0
    return float(np.percentile(a, q))


def _parse_area(area: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    """Parse --area 'latN,lonW,latS,lonE' → (n, w, s, e)."""
    if not area:
        return None
    parts = [p.strip() for p in str(area).split(",")]
    if len(parts) != 4:
        raise ValueError("--area must be 'latN,lonW,latS,lonE'")
    n, w, s, e = map(float, parts)
    return (n, w, s, e)


def _crop_aoi(ds: xr.Dataset, aoi: Tuple[float, float, float, float]) -> xr.Dataset:
    """Crop dataset to AOI; handles lon wrap if W>E (e.g., across anti-meridian)."""
    n, w, s, e = aoi  # north, west, south, east
    # Ensure lat is ascending
    ds = ds.sortby("lat")
    # Latitude slice (min..max)
    lo_lat, hi_lat = (min(s, n), max(s, n))
    ds = ds.sel(lat=slice(lo_lat, hi_lat))
    # Longitude slice (may wrap)
    if w <= e:
        ds = ds.sel(lon=slice(w, e))
    else:
        left = ds.sel(lon=slice(w, float(ds["lon"].max())))
        right = ds.sel(lon=slice(float(ds["lon"].min()), e))
        ds = xr.concat([left, right], dim="lon").sortby("lon")
    return ds


# ---------- core feature math ----------
def _vort_div(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    u, v shape: (T, Y, X)
    lat shape: (Y,), lon shape: (X,)
    crude finite-difference on sphere (meters) using separable gradients.
    """
    R = 6_371_000.0
    dlat_m = np.deg2rad(np.gradient(lat)) * R            # (Y,)
    dlon_r = np.deg2rad(np.gradient(lon))                # (X,)
    coslat = np.cos(np.deg2rad(lat))[:, None]            # (Y,1)
    dx_m = (dlon_r[None, :] * R) * coslat                # (Y,X)
    dy_m = dlat_m[:, None]                               # (Y,X)

    dx_m = np.where(dx_m == 0, np.nan, dx_m)
    dy_m = np.where(dy_m == 0, np.nan, dy_m)

    dudx = np.gradient(u, axis=-1) / dx_m
    dudy = np.gradient(u, axis=-2) / dy_m
    dvdx = np.gradient(v, axis=-1) / dx_m
    dvdy = np.gradient(v, axis=-2) / dy_m

    zeta = dvdx - dudy
    div  = dudx + dvdy
    return zeta.astype("float32"), div.astype("float32")


def _spiral_index(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    mag = np.sqrt(u*u + v*v).astype("float32")
    p99 = _safe_percentile(mag, 99.0) + 1e-8
    return (mag / p99).astype("float32")


def _relax_ratio(zeta: np.ndarray, div: np.ndarray) -> np.ndarray:
    eps = 1e-8
    return ((np.abs(zeta) + eps) / (np.abs(zeta) + np.abs(div) + eps)).astype("float32")


def _agreement(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    # 1 - normalized local variance in a cheap 5-point stencil
    def smooth(a):
        ap = np.pad(a, ((0,0),(1,1),(1,1)), mode="edge")
        return (a + ap[:,1:-1,:-2] + ap[:,1:-1,2:] + ap[:,:-2,1:-1] + ap[:,2:,1:-1]) / 5.0
    um, vm = smooth(u), smooth(v)
    var = (u - um)**2 + (v - vm)**2
    p99 = _safe_percentile(var, 99.0) + 1e-8
    out = 1.0 - np.clip(var / p99, 0, 1)
    return out.astype("float32")


def _msl_grad(msl: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    R = 6_371_000.0
    dlat_m = np.deg2rad(np.gradient(lat)) * R
    dlon_r = np.deg2rad(np.gradient(lon))
    coslat = np.cos(np.deg2rad(lat))[:, None]
    dx_m = (dlon_r[None, :] * R) * coslat
    dy_m = dlat_m[:, None]

    dx_m = np.where(dx_m == 0, np.nan, dx_m)
    dy_m = np.where(dy_m == 0, np.nan, dy_m)

    gx = np.gradient(msl, axis=-1) / dx_m
    gy = np.gradient(msl, axis=-2) / dy_m
    return np.sqrt(gx*gx + gy*gy).astype("float32")


def _delta_t(x: np.ndarray, dh: int) -> np.ndarray:
    """Simple forward difference with a dh-step (time axis 0)."""
    out = np.full_like(x, np.nan, dtype="float32")
    if dh <= 0 or x.shape[0] <= dh:
        return out
    out[dh:] = (x[dh:] - x[:-dh]).astype("float32")
    return out


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Build grid features from ERA5-like NetCDF (file/folder/glob robust).")
    ap.add_argument("--nc", action="append", help="Path to a .nc file or a folder (can repeat).")
    ap.add_argument("--nc-glob", help="Glob for .nc, supports ** (e.g., data_era5/monthly_nc/**/*stepType-*.nc)")
    ap.add_argument("--out-features", required=True, help="Output CSV/GZ or Parquet")
    ap.add_argument("--stride", type=int, default=8, help="Subsample stride over (lat,lon).")
    ap.add_argument("--delta-hours", type=int, default=6, help="Temporal delta step (hours).")

    # Lon normalization (new, aligned) + deprecated alias
    ap.add_argument("--normalize-lon", choices=("none","-180..180","0..360"), default="none",
                    help="Normalize longitude frame for output coords.")
    ap.add_argument("--lon-mode", choices=("auto","180","360"), default=None,
                    help="DEPRECATED alias: auto≈none, 180≈-180..180, 360≈0..360")

    ap.add_argument("--area", default=None, help="Crop AOI 'latN,lonW,latS,lonE' (optional).")
    # Optional variable name overrides
    ap.add_argument("--uvar", default=None, help="Name of 10m u-wind (default tries u10, 10m_u_component_of_wind)")
    ap.add_argument("--vvar", default=None, help="Name of 10m v-wind (default tries v10, 10m_v_component_of_wind)")
    ap.add_argument("--mslvar", default=None, help="Name of MSLP (default tries msl, mean_sea_level_pressure, sp)")
    # Optional coord name hints
    ap.add_argument("--time-name", default=None)
    ap.add_argument("--lat-name", default=None)
    ap.add_argument("--lon-name", default=None)
    args = ap.parse_args()

    # Map deprecated --lon-mode if used
    if args.lon_mode and args.normalize_lon == "none":
        args.normalize_lon = {"auto":"none", "180":"-180..180", "360":"0..360"}[args.lon_mode]

    files = _expand_inputs(args.nc or [], args.nc_glob)
    if not files:
        raise FileNotFoundError("No .nc files found. Use --nc, --nc-glob, or point --nc to a directory with .nc files.")
    print(f"[features] Using {len(files)} files (first 3): {files[:3]}")

    ds = _open_mfdataset_safely(files)
    ds, tname, latn, lonn = _standardize_axes(ds, args.time_name, args.lat_name, args.lon_name, args.normalize_lon)

    # Crop AOI if requested
    aoi = _parse_area(args.area)
    if aoi is not None:
        ds = _crop_aoi(ds, aoi)

    # Report bounds (post-normalization/crop)
    tvals = pd.to_datetime(ds[tname].values, utc=True)
    tmin = tvals.min().tz_convert("UTC").tz_localize(None)
    tmax = tvals.max().tz_convert("UTC").tz_localize(None)
    lon_min, lon_max = float(ds[lonn].min()), float(ds[lonn].max())
    lat_min, lat_max = float(ds[latn].min()), float(ds[latn].max())
    print(f"[features] TIME: {tmin} → {tmax}")
    print(f"[features] LAT : {lat_min:.3f} → {lat_max:.3f}   LON({args.normalize_lon}) : {lon_min:.3f} → {lon_max:.3f}")

    # Choose variable names
    def pick(name: Optional[str], cands: List[str]) -> str:
        if name and name in ds.variables: return name
        for c in cands:
            if c in ds.variables:
                return c
        raise ValueError(f"Could not find any of {cands} in dataset variables.")

    uvar = pick(args.uvar, ["u10", "10m_u_component_of_wind"])
    vvar = pick(args.vvar, ["v10", "10m_v_component_of_wind"])
    # Accept sp as fallback if msl missing (close enough for gradient structure)
    mvar = pick(args.mslvar, ["msl", "mean_sea_level_pressure", "sp", "surface_pressure"])

    # Align dims (T,Y,X) and pull to numpy as float32
    u10 = ds[uvar].transpose(tname, latn, lonn).astype("float32").values
    v10 = ds[vvar].transpose(tname, latn, lonn).astype("float32").values
    msl = ds[mvar].transpose(tname, latn, lonn).astype("float32").values
    # Time to tz-naive UTC
    time = pd.to_datetime(ds[tname].values, utc=True).tz_localize(None)
    lat  = ds[latn].values
    lon  = ds[lonn].values

    T, Y, X = u10.shape
    print(f"[features] Shape T={T}  Y={Y}  X={X}")

    # Core features
    zeta, div = _vort_div(u10, v10, lat, lon)
    S         = _spiral_index(u10, v10)
    relax     = _relax_ratio(zeta, div)
    agree     = _agreement(u10, v10)
    gradp     = _msl_grad(msl, lat, lon)

    # Temporal deltas
    dh = int(args.delta_hours)
    dS      = _delta_t(S, dh)
    drelax  = _delta_t(relax, dh)
    dagree  = _delta_t(agree, dh)

    # Subsample spatially
    yi = np.arange(0, Y, max(1, args.stride))
    xi = np.arange(0, X, max(1, args.stride))

    # Build rows (keeps memory steady and easy to read CSV/Parquet)
    rows = []
    for ti in range(T):
        tstamp = pd.to_datetime(time[ti])
        lat_slice = lat[yi]
        lon_slice = lon[xi]
        for y_idx, lat_y in zip(yi, lat_slice):
            for x_idx, lon_x in zip(xi, lon_slice):
                rows.append({
                    "time": tstamp,
                    "lat": float(lat_y),
                    "lon": float(lon_x),
                    "S": float(S[ti, y_idx, x_idx]),
                    "zeta": float(zeta[ti, y_idx, x_idx]),
                    "div":  float(div[ti, y_idx, x_idx]),
                    "relax":     float(relax[ti, y_idx, x_idx]),
                    "agree":     float(agree[ti, y_idx, x_idx]),
                    "dS_dt":     float(dS[ti, y_idx, x_idx]) if ti >= dh else np.nan,
                    "drelax_dt": float(drelax[ti, y_idx, x_idx]) if ti >= dh else np.nan,
                    "dagree_dt": float(dagree[ti, y_idx, x_idx]) if ti >= dh else np.nan,
                    "msl_grad":  float(gradp[ti, y_idx, x_idx]),
                })

    df = pd.DataFrame(rows)

    # Optional hook for user-defined extra features
    df = _maybe_apply_user_enhancements(df, ds)

    # Clean + types
    for c in ("lat", "lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["time", "lat", "lon"]).reset_index(drop=True)

    # Write (CSV/GZ or Parquet inferred from extension)
    _write_any(args.out_features, df)

    hours = df["time"].dt.floor("h").nunique()
    print(f"[features] Wrote {args.out_features}  rows={len(df):,}  hours={hours}  "
          f"grid≈{len(yi)}×{len(xi)} (stride={args.stride})")


if __name__ == "__main__":
    main()