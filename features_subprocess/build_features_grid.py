#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_features_grid.py
Flatten ERA5 NetCDFs into a tidy grid feature table:
time (UTC, tz-naive), lat, lon, and chosen variables (+ optional derived wspd, zeta, div, S, agree).

Fixes / Enhancements:
  • Accepts --out and --out-features (aliases).
  • Opt-in keeping of u/v via --export-uv.
  • --force-keep to pin variables that must be retained if present.
  • Robust variable binding via alias lists (e.g., 10m_u_component_of_wind ↔ u10).
  • Optional --engine and multi-engine fallback (netcdf4 -> h5netcdf -> scipy).
  • Accept normalize-lon values with or without leading spaces (argparse quirk).
  • MSL auto-convert Pa->hPa if median suggests Pascals.
  • --require-vars accepts CSV, space-separated, or repeated flags (friendly to YAML flattening).
  • On first skip from missing required vars, prints a sample of present variables.
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

def _fix_normalize_lon_tokens(argv: list[str]) -> list[str]:
    """
    Allow --normalize-lon values that start with '-' to be passed without quoting by
    rewriting '--normalize-lon -180..180' -> '--normalize-lon=-180..180'. Also trims
    stray whitespace in the value token.
    """
    fixed: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("--normalize-lon", "--normalize_lon") and i + 1 < len(argv):
            val_raw = argv[i + 1]
            val = val_raw.strip()
            if val:
                fixed.append(f"{tok}={val}")
                i += 2
                continue
        fixed.append(tok)
        i += 1
    return fixed


def parse_args(argv: list[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = _fix_normalize_lon_tokens(argv)

    ap = argparse.ArgumentParser(description="Flatten ERA5 NetCDFs to grid features (time, lat, lon, vars).")
    # Inputs
    ap.add_argument("--nc-glob", default=None, help="Glob of NetCDFs (e.g. data/**/*.nc)")
    ap.add_argument("--nc", action="append", default=None, help="Add one or more explicit .nc paths (repeatable)")
    ap.add_argument("--engine", default=None, help="xarray engine hint (netcdf4, h5netcdf, scipy, cfgrib)")

    # Output (aliases)
    out_group = ap.add_mutually_exclusive_group(required=True)
    out_group.add_argument("--out", dest="out_features", help="Output CSV(.gz) or Parquet path")
    out_group.add_argument("--out-features", dest="out_features", help="Output CSV(.gz) or Parquet path (alias)")

    # Variable names (accept aliases; see ALIASES map below)
    ap.add_argument("--uvar", default=None, help="u-component (e.g., u10 or 10m_u_component_of_wind)")
    ap.add_argument("--vvar", default=None, help="v-component (e.g., v10 or 10m_v_component_of_wind)")
    ap.add_argument("--mslvar", default=None, help="mean sea level pressure var (e.g., msl or mean_sea_level_pressure)")
    ap.add_argument("--t2mvar", default=None, help="2m temperature var (e.g., t2m or 2m_temperature)")

    ap.add_argument(
        "--require-vars",
        nargs="*",
        default=None,
        help="Require ALL (alias-aware). Accepts CSV or space-separated "
             "(e.g., u10 v10 msl t2m OR 'u10,v10,msl,t2m').",
    )
    ap.add_argument(
        "--force-keep",
        nargs="*",
        default=None,
        help="Optional list of variables to keep if present (e.g., cape cin blh).",
    )

    # Coord names
    ap.add_argument("--time-name", default=None)
    ap.add_argument("--lat-name",  default=None)
    ap.add_argument("--lon-name",  default=None)

    # Domain / thinning
    ap.add_argument(
        "--normalize-lon",
        default="none",
        nargs="?",
        const="-180..180",
    )
    ap.add_argument("--area", default=None, help="latN,lonW,latS,lonE  (match lon range to normalize-lon)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--delta-hours", type=int, default=1)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end",   default=None)
    # Chunking knobs (accepted for orchestrator compatibility; not used in this script)
    ap.add_argument("--chunk-rows", type=int, default=0, help="Accepted for compatibility; unused here.")
    ap.add_argument("--chunksize", type=int, default=0, help="Accepted for compatibility; unused here.")
    ap.add_argument("--parquet-rows", type=int, default=0, help="Accepted for compatibility; unused here.")
    # Agent: accept overwrite flag for pipeline compatibility (output overwrites by default).
    ap.add_argument("--overwrite", action="store_true", help="No-op; output is overwritten if present.")

    # Grid identity & duplicates
    ap.add_argument("--emit-grid-index", action="store_true")
    ap.add_argument("--dedup", choices=["none", "time_lat_lon"], default="none")

    # Derived fields
    ap.add_argument(
        "--with-vortdiv",
        action="store_true",
        help="Compute vorticity/divergence per grid cell and derived proxies S, agree.",
    )
    ap.add_argument(
        "--export-uv",
        action="store_true",
        help="Include raw u and v wind components in the output frame.",
    )

    args = ap.parse_args()

    # Normalize --require-vars: supports CSV, whitespace, repeated flags.
    if args.require_vars:
        norm: list[str] = []
        for tok in args.require_vars:
            if tok is None:
                continue
            parts = [p.strip() for p in tok.replace(",", " ").split() if p.strip()]
            norm.extend(parts)
        args.require_vars = norm if norm else None

    return args

# ---------------- alias binding ----------------

ALIASES = {
    "u10": ["u10", "10m_u_component_of_wind", "U10M", "u_10m"],
    "v10": ["v10", "10m_v_component_of_wind", "V10M", "v_10m"],
    "msl": ["msl", "mean_sea_level_pressure", "MSL", "prmsl"],
    "t2m": ["t2m", "2m_temperature", "T2M", "t_2m"],
}

def _resolve_alias(name: str | None, present: set[str], key: str | None = None) -> str | None:
    """Try explicit name, then alias group keyed by `key`."""
    if name and name in present:
        return name
    if key and key in ALIASES:
        # direct alias search
        for cand in ALIASES[key]:
            if cand in present:
                return cand
        # if user passed one alias name, allow any alias that is present
        if name and name in ALIASES[key]:
            for cand in ALIASES[key]:
                if cand in present:
                    return cand
    return None

def _resolve_require_vars(require_list: list[str] | None, present: set[str]) -> bool:
    """Alias-aware presence check for --require-vars (now normalized)."""
    if not require_list:
        return True
    for r in require_list:
        if r in present:
            continue
        found = False
        for _, cands in ALIASES.items():
            if r in cands:
                if any(c in present for c in cands):
                    found = True
                    break
        if not found:
            return False
    return True

def _missing_required(require_list: list[str] | None, present: set[str]) -> list[str]:
    """Return the subset of require_list that are absent (alias-aware)."""
    if not require_list:
        return []
    missing: list[str] = []
    for r in require_list:
        if r in present:
            continue
        found = False
        for _, cands in ALIASES.items():
            if r in cands and any(c in present for c in cands):
                found = True
                break
        if not found:
            missing.append(r)
    return missing

# ---------------- coord normalization ----------------

POSSIBLE_TIME_NAMES = ("time", "valid_time", "forecast_reference_time")
POSSIBLE_LAT_NAMES  = ("lat", "latitude", "Latitude", "nav_lat")
POSSIBLE_LON_NAMES  = ("lon", "longitude", "Longitude", "nav_lon")

def _pick_name(cands, present):
    for c in cands:
        if c in present:
            return c
    return None

def collapse_expver(ds: xr.Dataset) -> xr.Dataset:
    """Collapse ECMWF expver dimension into a single best-estimate field."""
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
    if ren:
        ds = ds.rename(ren)

    for c in ("time", "lat", "lon"):
        if c in ds and c not in ds.coords:
            ds = ds.set_coords(c)

    return ds

# ---------------- lon reframing / area ----------------

def _canon_norm(mode: str) -> str:
    return (mode or "none").strip()

def reframe_lon_vals(lon_vals: np.ndarray, mode: str) -> np.ndarray:
    mode = _canon_norm(mode)
    if mode == "none":
        return lon_vals
    if mode == "0..360":
        return (lon_vals % 360 + 360) % 360
    # default: -180..180
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

# ---------------- thinning utilities ----------------

def keep_delta_hours(df: pd.DataFrame, delta: int) -> pd.DataFrame:
    if delta is None or delta <= 1:
        return df
    t = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    return df.loc[(t.dt.hour.to_numpy() % int(delta)) == 0].copy()

def apply_area_df(df: pd.DataFrame, aoi):
    if not aoi:
        return df
    latN, lonW, latS, lonE = aoi
    return df.loc[
        (df["lat"] <= latN) & (df["lat"] >= latS) &
        (df["lon"] >= lonW) & (df["lon"] <= lonE)
    ].copy()

def stride_df(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    if stride <= 1 or df.empty:
        return df

    df = df.sort_values(["time", "lat", "lon"], kind="mergesort").reset_index(drop=True)

    def _per_time(g):
        lats = np.sort(g["lat"].unique())
        lons = np.sort(g["lon"].unique())
        lat_map = {v: i for i, v in enumerate(lats)}
        lon_map = {v: i for i, v in enumerate(lons)}
        gi = g.copy()
        gi["_ilat"] = g["lat"].map(lat_map).to_numpy()
        gi["_ilon"] = g["lon"].map(lon_map).to_numpy()
        keep = (gi["_ilat"] % stride == 0) & (gi["_ilon"] % stride == 0)
        return gi.loc[keep].drop(columns=["_ilat", "_ilon"])

    return df.groupby(
        pd.to_datetime(df["time"]).dt.floor("h"),
        sort=False,
        group_keys=False
    ).apply(_per_time)

# ---------------- vorticity/divergence helpers ----------------

def compute_zeta_div(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    """Latitude-aware vorticity (zeta) and divergence (div) in s^-1 (approx on regular lat/lon)."""
    m_per_deg_y = 110_540.0
    m_per_deg_x_row = 111_320.0 * np.cos(np.deg2rad(lat))
    dlat_deg = np.gradient(lat)
    dlon_deg = np.gradient(lon)

    y_m = np.cumsum(np.r_[0.0, m_per_deg_y * dlat_deg[1:]])
    x_m_nominal = np.cumsum(np.r_[0.0, (m_per_deg_x_row[0] * dlon_deg[1:])])

    dU_dy, dU_dx = np.gradient(u, y_m, x_m_nominal, edge_order=1)
    dV_dy, dV_dx = np.gradient(v, y_m, x_m_nominal, edge_order=1)

    # Adjust dx scaling away from equator row
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

    # Downcast only the actual data variables; coords are left alone so joins remain exact.
    for v in list(sub.data_vars):
        if np.issubdtype(sub[v].dtype, np.floating):
            sub[v] = sub[v].astype("float32")

    df = sub.to_dataframe().reset_index()

    for c in ("time", "lat", "lon"):
        if c not in df.columns:
            if c in df.index.names:
                df = df.reset_index(c)
            else:
                raise KeyError(f"Missing '{c}' column after to_dataframe(); columns={list(df.columns)}")

    return df.dropna(subset=["time", "lat", "lon"])

# ---------------- open helpers ----------------

def _open_dataset_with_fallback(path: str, hint_engine: str | None):
    engines = [hint_engine] if hint_engine else []
    engines += ["netcdf4", "h5netcdf", "scipy"]
    tried = []

    for eng in engines:
        if eng is None:
            continue
        try:
            return xr.open_dataset(path, engine=eng)
        except Exception as e:
            tried.append((eng, str(e)[:120]))

    try:
        return xr.open_dataset(path)
    except Exception as e:
        msg = "; ".join([f"{eng}:{err}" for eng, err in tried]) or "no engines tried"
        raise RuntimeError(f"open_dataset failed for {path}. Tried -> {msg}. Last error: {e}")

# ---------------- main ----------------

def main():
    args = parse_args()

    files: list[str] = []
    if args.nc_glob:
        files += glob.glob(args.nc_glob, recursive=True)
    if args.nc:
        files += [p for p in args.nc if p]

    files = sorted({str(f) for f in files if Path(f).exists()})
    if not files:
        print("[err] No NetCDF files found.", file=sys.stderr)
        sys.exit(2)

    force_keep = set(args.force_keep or [])
    required_raw = list(args.require_vars or [])

    area_box = parse_area(args.area)
    out_path = Path(args.out_features)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dfs: list[pd.DataFrame] = []
    ok_files = 0
    skipped_require = 0
    printed_missing_debug = False

    for i, f in enumerate(files, 1):
        ds = _open_dataset_with_fallback(f, args.engine)
        ds = normalize_coords(ds, args.time_name, args.lat_name, args.lon_name)
        ds = reframe_lon_ds(ds, args.normalize_lon)
        ds = select_aoi_ds(ds, area_box)

        present_vars = set(ds.data_vars)

        # Required vars check (alias-aware) -> fail fast with diagnosis
        missing_required = _missing_required(required_raw, present_vars)
        if missing_required:
            sample = ""
            if not printed_missing_debug:
                printed_missing_debug = True
                try:
                    pv = sorted(list(present_vars))
                    extra = max(0, len(pv) - 40)
                    sample = f" Present variables (sample): {pv[:40]}" + (f" +{extra} more" if extra > 0 else "")
                except Exception:
                    sample = ""
            ds.close()
            raise SystemExit(
                f"[err] {Path(f).name}: missing required variables {missing_required}. "
                f"Upstream ERA5 merge likely dropped them.{sample}"
            )

        # Resolve aliases for core vars
        u_name   = _resolve_alias(args.uvar,   present_vars, "u10")
        v_name   = _resolve_alias(args.vvar,   present_vars, "v10")
        msl_name = _resolve_alias(args.mslvar, present_vars, "msl")
        t2m_name = _resolve_alias(args.t2mvar, present_vars, "t2m")

        keep_vars: list[str] = []

        # Winds and derived fields
        if u_name and v_name:
            U = ds[u_name].astype("float32")
            V = ds[v_name].astype("float32")

            if args.export_uv:
                ds = ds.assign({u_name: U, v_name: V})
                keep_vars += [u_name, v_name]

            wspd = np.sqrt(U**2 + V**2).astype("float32")
            ds = ds.assign(wspd=wspd)
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

                zeta_da = xr.DataArray(
                    np.stack(z_stack, axis=0),
                    coords={"time": ds["time"], "lat": ds["lat"], "lon": ds["lon"]},
                    dims=("time", "lat", "lon"),
                    name="zeta",
                ).astype("float32")
                div_da = xr.DataArray(
                    np.stack(d_stack, axis=0),
                    coords={"time": ds["time"], "lat": ds["lat"], "lon": ds["lon"]},
                    dims=("time", "lat", "lon"),
                    name="div",
                ).astype("float32")

                ds = ds.assign(zeta=zeta_da, div=div_da)
                keep_vars += ["zeta", "div"]

                S = np.sqrt(zeta_da**2 + div_da**2).astype("float32")
                ds = ds.assign(S=S)
                keep_vars.append("S")

                agree = (np.abs(zeta_da) > np.abs(div_da)).astype("float32")
                ds = ds.assign(agree=agree)
                keep_vars.append("agree")

        # Mean sea-level pressure (Pa->hPa if needed)
        if msl_name:
            msl_da = ds[msl_name].astype("float32")
            try:
                med = float(np.nanmedian(msl_da.values))
                if med > 2000.0:
                    msl_da = msl_da / 100.0
            except Exception:
                pass
            ds = ds.assign(msl=msl_da)
            keep_vars.append("msl")

        # 2m temperature (keep as-is, in K)
        if t2m_name:
            ds = ds.assign(t2m=ds[t2m_name].astype("float32"))
            keep_vars.append("t2m")

        # Force-keep extras if present
        for v in list(force_keep):
            if v in present_vars:
                dv = ds[v]
                if np.issubdtype(dv.dtype, np.floating):
                    dv = dv.astype("float32")
                ds = ds.assign({v: dv})
                keep_vars.append(v)

        # Default fallback: keep all numeric data variables
        if not keep_vars:
            keep_vars = [k for k in ds.data_vars if np.issubdtype(ds[k].dtype, np.number)]

        try:
            df = to_frame(ds, keep_vars)
        except Exception as e:
            print(f"[skip {i}] {Path(f).name}: to_frame error {e}")
            ds.close()
            continue

        # Time filters and geo post-processing
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
        if args.start:
            t0 = pd.to_datetime(args.start)
            df = df[df["time"] >= t0]
        if args.end:
            t1 = pd.to_datetime(args.end)
            df = df[df["time"] < t1]
        if args.delta_hours and args.delta_hours > 1:
            print(
                f"[warn] delta_hours={args.delta_hours} -> thinning times; "
                "for slow-tick/GSE training use delta_hours=1.",
                file=sys.stderr,
            )
        df = keep_delta_hours(df, args.delta_hours)
        df = apply_area_df(df, area_box)

        if _canon_norm(args.normalize_lon) != "none":
            df["lon"] = reframe_lon_vals(
                pd.to_numeric(df["lon"], errors="coerce").to_numpy(),
                _canon_norm(args.normalize_lon),
            )

        if args.stride > 1:
            df = stride_df(df, args.stride)

        # Normalize time to tz-naive UTC
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=["time", "lat", "lon"]).reset_index(drop=True)

        if not df.empty:
            dfs.append(df)
            ok_files += 1
            kept = sorted(set(df.columns) - {"time", "lat", "lon"})
            print(f"[{i}/{len(files)}] {Path(f).name}: rows={len(df):,}  vars={kept}")

        ds.close()

    if not dfs:
        print(
            f"[err] No rows produced. Files read={ok_files}, "
            f"skipped_missing_required={skipped_require}",
            file=sys.stderr,
        )
        sys.exit(1)

    out_df = pd.concat(dfs, ignore_index=True)

    if args.dedup == "time_lat_lon":
        out_df = out_df.drop_duplicates(subset=["time", "lat", "lon"], keep="first", ignore_index=True)

    if args.emit_grid_index and not out_df.empty:
        lats = np.sort(out_df["lat"].unique())
        lons = np.sort(out_df["lon"].unique())
        out_df["ilat"] = out_df["lat"].map({v: i for i, v in enumerate(lats)}).astype("int32")
        out_df["ilon"] = out_df["lon"].map({v: i for i, v in enumerate(lons)}).astype("int32")

    out_df.sort_values(["time", "lat", "lon"], inplace=True, ignore_index=True)

    out_path = Path(args.out_features)
    name_low = out_path.name.lower()
    if name_low.endswith((".parquet", ".parq", ".pq")):
        out_df.to_parquet(out_path, index=False)
    else:
        comp = "gzip" if (name_low.endswith(".csv.gz") or out_path.suffix.lower() == ".gz") else "infer"
        out_df.to_csv(out_path, index=False, compression=comp, date_format="%Y-%m-%d %H:%M:%S")

    H = out_df["lat"].nunique()
    W = out_df["lon"].nunique()
    T = out_df["time"].nunique()
    print(f"[ok] wrote {len(out_df):,} rows -> {out_path}  (H={H} x W={W} x T={T})")
    if args.emit_grid_index:
        print("  (ilat/ilon present -> stable grid IDs across hours)")

if __name__ == "__main__":
    main()
