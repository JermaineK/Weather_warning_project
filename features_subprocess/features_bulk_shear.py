#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
features_bulk_shear.py
Compute vertical bulk-shear magnitudes from ERA5 pressure-level wind NetCDFs.

Robust to:
  - Different level coord names (auto-detects non-{time,lat,lon} shared dim)
  - Level units in Pa or hPa (auto-normalizes to hPa for matching)
  - u/v variable aliases (overrideable)

Outputs columns: time, lat, lon, shear_low, shear_deep, S3

Examples:
  python features_bulk_shear.py \
    --pl-glob "data_era5/extracted/**/era5_pl_*_uv.nc" \
    --out data/features_bulk_shear.parquet \
    --low-pair 1000,925 --deep-pair 1000,500 --s3-window 3 --s3-source deep \
    --normalize-lon -180..180 --area "-5,125,-35,175" --overwrite
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
from typing import Tuple, Optional, Sequence
import numpy as np
import pandas as pd
import xarray as xr

# ---------------- util: tolerant open & coord standardization ----------------

def _open_mf(paths: Sequence[str]) -> xr.Dataset:
    errs = []
    for eng in (None, "netcdf4", "h5netcdf"):
        try:
            return xr.open_mfdataset(paths, combine="by_coords", engine=eng)
        except Exception as e:
            errs.append((eng or "default", str(e)))
    msg = " | ".join(f"{k}: {v}" for k, v in errs[:2]) + (" ..." if len(errs) > 2 else "")
    raise RuntimeError(f"open_mfdataset failed for all engines. First errors: {msg}")

def _std_coords(ds: xr.Dataset) -> xr.Dataset:
    # time
    if "time" not in ds.coords and "time" in ds.variables:
        ds = ds.set_coords("time")
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    # lat/lon renames
    if "latitude" in ds.coords and "lat" not in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.coords and "lon" not in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    for c in ("time", "lat", "lon"):
        if c in ds and c not in ds.coords:
            ds = ds.set_coords(c)
    # wrap lon to [-180,180)
    if "lon" in ds:
        ds = ds.assign_coords(lon=((ds["lon"] + 180) % 360) - 180).sortby("lon")
    # ensure ascending lat
    if "lat" in ds:
        la = np.asarray(ds["lat"].values)
        if la.size > 1 and (la[1] - la[0]) < 0:
            ds = ds.sortby("lat")
    return ds

# ---------------- u/v & level detection ----------------

UV_ALIASES = [
    ("u", "v"),
    ("U", "V"),
    ("u_component_of_wind", "v_component_of_wind"),
    ("eastward_wind", "northward_wind"),
]

def _bind_uv(ds: xr.Dataset, u_hint: Optional[str], v_hint: Optional[str]) -> Tuple[str, str]:
    if u_hint and v_hint:
        if u_hint in ds.data_vars and v_hint in ds.data_vars:
            return u_hint, v_hint
    for u_name, v_name in UV_ALIASES:
        if u_name in ds.data_vars and v_name in ds.data_vars:
            return u_name, v_name
    # fallbacks: first two vector-looking names?
    cand = [k for k in ds.data_vars if k.lower().startswith(("u", "v", "east", "north"))]
    if len(cand) >= 2:
        return cand[0], cand[1]
    raise KeyError(f"Could not find u/v variables. Present: {list(ds.data_vars)[:10]}")

CANON_AXES = {"time", "lat", "lon"}

def _detect_level_dim(ds: xr.Dataset, u_name: str, v_name: str) -> str:
    # a level dim is a non-{time,lat,lon} dimension shared by both u and v and length>1
    u_dims = set(ds[u_name].dims)
    v_dims = set(ds[v_name].dims)
    shared = (u_dims & v_dims) - CANON_AXES
    # prefer dims with a matching coord var present
    candidates = [d for d in shared if d in ds.coords or d in ds.variables]
    # final list, most promising first (keep stable order from u dims)
    ordered = [d for d in ds[u_name].dims if d in candidates]
    if not ordered:
        # fallback: any shared non-canon dim
        ordered = [d for d in ds[u_name].dims if d in shared]
    if not ordered:
        raise KeyError(f"No pressure-level coord found. Shared dims={shared}  u.dims={ds[u_name].dims} v.dims={ds[v_name].dims}")
    return ordered[0]

def _levels_hpa(ds: xr.Dataset, lvl_name: str) -> np.ndarray:
    """Return level values in hPa (convert from Pa if needed)."""
    if lvl_name in ds.coords:
        vals = np.array(ds.coords[lvl_name].values).astype(float)
    elif lvl_name in ds.variables:
        vals = np.array(ds.variables[lvl_name].values).astype(float)
    else:
        # dim exists but no coord var — synthesize 0..N-1 (not useful). Bail out.
        raise KeyError(f"Level dimension '{lvl_name}' has no coordinate variable.")
    # unit sniff
    units = (ds.coords.get(lvl_name, ds.variables.get(lvl_name))).attrs.get("units", "").lower()
    if "pa" in units and "hpa" not in units:
        # looks like Pa → convert
        vals = vals / 100.0
    else:
        # heuristic: big numbers mean Pa
        if np.nanmax(vals) > 3000:  # 100000 Pa, etc.
            vals = vals / 100.0
    return vals

def _nearest_levels_indices(levels_hpa: np.ndarray, a_hpa: float, b_hpa: float) -> Tuple[int, int]:
    a_idx = int(np.nanargmin(np.abs(levels_hpa - float(a_hpa))))
    b_idx = int(np.nanargmin(np.abs(levels_hpa - float(b_hpa))))
    return a_idx, b_idx

# ---------------- math ----------------

def _bulk_mag(u2: xr.DataArray, v2: xr.DataArray, lvl_name: str, i0: int, i1: int) -> xr.DataArray:
    # ensure lvl axis order is consistent
    u2 = u2.transpose(..., lvl_name, missing_dims="ignore")
    v2 = v2.transpose(..., lvl_name, missing_dims="ignore")
    # pick two slices and compute |ΔV|
    du = u2.isel({lvl_name: i1}) - u2.isel({lvl_name: i0})
    dv = v2.isel({lvl_name: i1}) - v2.isel({lvl_name: i0})
    return np.hypot(du, dv).astype("float32")

def _to_naive_utc(s):
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None)

def _wrap_lon(x, mode):
    arr = pd.to_numeric(x, errors="coerce").to_numpy()
    m = str(mode or "none").replace(" ", "")
    if m == "0..360": return (arr % 360 + 360) % 360
    if m in ("-180..180", "-180…180"): return ((arr + 180) % 360) - 180
    return arr

# ---------------- IO helpers ----------------

def _write_any(path: str, df: pd.DataFrame, overwrite=True):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not overwrite:
        print(f"[skip] exists: {p}")
        return
    low = p.name.lower()
    if low.endswith((".parquet", ".parq", ".pq", ".pqt")):
        df.to_parquet(p, index=False)
    else:
        comp = "gzip" if (low.endswith(".gz") or p.suffix.lower() == ".gz") else "infer"
        df.to_csv(p, index=False, compression=comp, date_format="%Y-%m-%d %H:%M:%S")

# ---------------- CLI ----------------

def parse_args():
    ap = argparse.ArgumentParser(description="Bulk shear from pressure-level u/v (robust level detection)")
    ap.add_argument("--pl-glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--low-pair", required=True, help="e.g. 1000,925  (hPa)")
    ap.add_argument("--deep-pair", required=True, help="e.g. 1000,500 (hPa)")
    ap.add_argument("--s3-window", type=int, default=3, help="rolling window in hours for S3")
    ap.add_argument("--s3-source", choices=["deep", "low"], default="deep", help="which shear to smooth for S3")
    ap.add_argument("--normalize-lon", default="-180..180")
    ap.add_argument("--area", default=None, help="latN,lonW,latS,lonE")
    ap.add_argument("--overwrite", action="store_true")

    # power-user overrides & introspection
    ap.add_argument("--u-name", default=None, help="explicit u variable name if not 'u'")
    ap.add_argument("--v-name", default=None, help="explicit v variable name if not 'v'")
    ap.add_argument("--level-name", default=None, help="explicit level dim/coord name")
    ap.add_argument("--list", action="store_true", help="print dataset summary (dims/coords/vars) and exit")
    return ap.parse_args()

# ---------------- main ----------------

def main():
    args = parse_args()
    files = sorted(glob.glob(args.pl_glob, recursive=True))
    if not files:
        print(f"[bulk-shear] no matches: {args.pl_glob}", file=sys.stderr); sys.exit(2)

    ds = _std_coords(_open_mf(files))

    # optional introspection
    if args.list:
        print("== Dataset summary ==")
        print("Dims :", dict(ds.dims))
        print("Coords:", list(ds.coords))
        print("Vars :", list(ds.data_vars)[:20])
        try: ds.close()
        except Exception: pass
        return

    # bind u/v (aliases or hints)
    u_name, v_name = _bind_uv(ds, args.u_name, args.v_name)

    # detect or use provided level dim
    if args.level_name:
        lvl_name = args.level_name
        if lvl_name not in ds[u_name].dims or lvl_name not in ds[v_name].dims:
            raise KeyError(f"Provided --level-name '{lvl_name}' is not a dim of u/v: u.dims={ds[u_name].dims}, v.dims={ds[v_name].dims}")
    else:
        lvl_name = _detect_level_dim(ds, u_name, v_name)

    # level values → hPa
    levels_hpa = _levels_hpa(ds, lvl_name)
    low_a, low_b   = [float(x) for x in str(args.low_pair).split(",")]
    deep_a, deep_b = [float(x) for x in str(args.deep_pair).split(",")]

    low_i0, low_i1     = _nearest_levels_indices(levels_hpa, low_a,  low_b)
    deep_i0, deep_i1   = _nearest_levels_indices(levels_hpa, deep_a, deep_b)

    # compute |ΔV|
    shear_low  = _bulk_mag(ds[u_name], ds[v_name], lvl_name, low_i0,  low_i1).rename("shear_low")
    shear_deep = _bulk_mag(ds[u_name], ds[v_name], lvl_name, deep_i0, deep_i1).rename("shear_deep")

    base = shear_deep if args.s3_source == "deep" else shear_low
    S3 = (base if args.s3_window <= 1 else base.rolling(time=int(args.s3_window), min_periods=1).mean()).rename("S3")

    out = xr.merge([shear_low, shear_deep, S3]).to_dataframe().reset_index()

    # tidy columns
    if "time" in out: out["time"] = _to_naive_utc(out["time"])
    if "lon" in out:  out["lon"]  = _wrap_lon(out["lon"], args.normalize_lon)

    # optional area crop
    if args.area:
        try:
            N, W, S, E = [float(x) for x in str(args.area).split(",")]
            out = out.loc[(out["lat"] >= S) & (out["lat"] <= N) & (out["lon"] >= W) & (out["lon"] <= E)]
        except Exception:
            pass

    _write_any(args.out, out, overwrite=args.overwrite)
    print(f"[bulk-shear] wrote {args.out} rows={len(out):,}  "
          f"(u='{u_name}', v='{v_name}', level='{lvl_name}', levels~{int(len(levels_hpa))} @hPa)")

    try:
        ds.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()