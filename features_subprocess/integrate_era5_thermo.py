#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integrate_era5_thermo.py  — streaming, low-memory integrator (v3)

- Streams features in batches (Parquet or CSV) and ERA5 single-level NetCDFs per file.
- Exact (time,lat,lon) merge or per-hour nearest-neighbor merge.
- Version-robust for pyarrow.dataset (no .scan()).
- Avoids Arrow→Pandas ExtensionArray pitfalls:
    * no types_mapper
    * no predeclared schema — writer created from first output batch
"""

from __future__ import annotations
import argparse, glob, sys, math, warnings
from pathlib import Path
from typing import Iterable, Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# Optional KDTree for nearest mode
try:
    from scipy.spatial import cKDTree
    HAVE_KDTREE = True
except Exception:
    HAVE_KDTREE = False

# ---------------- CLI ----------------

def parse_args():
    ap = argparse.ArgumentParser(description="Integrate ERA5 thermo onto features (streaming, low-memory).")
    ap.add_argument("--features", required=True, help="Features table (Parquet or CSV(.gz))")

    # prefer thermo_glob; accept nc_glob alias
    ap.add_argument("--thermo-glob", default=None, help="Glob of ERA5 single-level *.nc files")
    ap.add_argument("--nc-glob",     default=None, help="Alias of --thermo-glob")

    ap.add_argument("--out", required=True, help="Output: Parquet or CSV(.gz)")

    ap.add_argument("--normalize-lon",
                    choices=["none","-180..180","0..360"," -180..180"," 0..360"],
                    default="none")

    ap.add_argument("--vars", nargs="*", default=["u10","v10","msl","t2m"],
                    help="Thermo vars to extract and attach")

    ap.add_argument("--nearest", action="store_true",
                    help="Per-hour nearest-neighbor instead of exact (time,lat,lon) join")
    ap.add_argument("--nearest-maxdeg", type=float, default=0.4,
                    help="Angular threshold (deg) for nearest; outside → NaN")

    ap.add_argument("--row-group-rows", type=int, default=1_000_000,
                    help="Parquet row group size when writing")
    ap.add_argument("--scan-batch-rows", type=int, default=500_000,
                    help="Max rows per batch read from features (Parquet)")
    ap.add_argument("--csv-chunk-rows", type=int, default=1_000_000,
                    help="Rows per CSV write chunk")

    ap.add_argument("--engine", default=None, help="xarray engine hint")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()

# ---------------- lon helpers ----------------

def _canon_norm(mode: str) -> str:
    return (mode or "none").strip()

def reframe_lon_vals(lon_vals: np.ndarray, mode: str) -> np.ndarray:
    mode = _canon_norm(mode)
    if mode == "none":
        return lon_vals
    if mode == "0..360":
        return (lon_vals % 360 + 360) % 360
    return ((lon_vals + 180) % 360) - 180

# ---------------- NetCDF → DataFrame ----------------

def _open_dataset_with_fallback(path: str, hint: Optional[str]):
    engines = [hint] if hint else []
    engines += ["netcdf4", "h5netcdf", "scipy"]
    for eng in engines:
        if eng is None: continue
        try:
            return xr.open_dataset(path, engine=eng)
        except Exception:
            pass
    return xr.open_dataset(path)

def _nc_to_frame(nc_path: str, vars_wanted: List[str], norm_lon: str) -> pd.DataFrame:
    dsx = _open_dataset_with_fallback(nc_path, hint=None)

    def _pick(cands):
        for c in cands:
            if c in dsx.coords or c in dsx.dims:
                return c
        return None

    tname = _pick(("time","valid_time","forecast_reference_time"))
    latn  = _pick(("lat","latitude","Latitude","nav_lat"))
    lonn  = _pick(("lon","longitude","Longitude","nav_lon"))
    if not all([tname, latn, lonn]):
        raise ValueError(f"Missing coords in {nc_path}: time={tname}, lat={latn}, lon={lonn}")
    ren = {}
    if tname!="time": ren[tname]="time"
    if latn!="lat":   ren[latn]="lat"
    if lonn!="lon":   ren[lonn]="lon"
    if ren: dsx = dsx.rename(ren)

    # expver collapse
    if "expver" in dsx.dims:
        try:
            values = set(np.array(dsx.coords["expver"].values).tolist())
            if 1 in values:
                ds1 = dsx.sel(expver=1)
                if 5 in values:
                    ds5 = dsx.sel(expver=5)
                    dsx = ds1.combine_first(ds5)
                else:
                    dsx = ds1
            else:
                dsx = dsx.isel(expver=0)
        except Exception:
            dsx = dsx.isel(expver=0)
        dsx = dsx.squeeze(drop=True)

    # reframe longitudes and sort
    lon2 = reframe_lon_vals(dsx["lon"].to_numpy(), norm_lon)
    order = np.argsort(lon2)
    dsx = dsx.assign_coords(lon=("lon", lon2))
    if not np.all(order == np.arange(len(lon2))):
        dsx = dsx.sortby("lon")

    present = set(dsx.data_vars)
    alias_map = {
        "u10": ["u10","10m_u_component_of_wind","U10M","u_10m"],
        "v10": ["v10","10m_v_component_of_wind","V10M","v_10m"],
        "msl": ["msl","mean_sea_level_pressure","MSL","prmsl"],
        "t2m": ["t2m","2m_temperature","T2M","t_2m"],
    }
    keep: Dict[str,str] = {}
    for want in vars_wanted:
        if want in present:
            keep[want] = want
        elif want in alias_map:
            for a in alias_map[want]:
                if a in present:
                    keep[want] = a
                    break

    vars_real = list(set(keep.values()))
    if vars_real:
        sub = dsx[vars_real]
        rev = {v:k for k,v in keep.items()}
        sub = sub.rename({v: rev.get(v, v) for v in vars_real})
        df = sub.to_dataframe().reset_index()
    else:
        df = dsx[[]].to_dataframe().reset_index()

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    for c in vars_wanted:
        if c in df.columns and pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("float32")

    keep_cols = ["time","lat","lon"] + [c for c in vars_wanted if c in df.columns]
    df = df[keep_cols].dropna(subset=["time","lat","lon"]).reset_index(drop=True)
    dsx.close()
    return df

def _time_bounds(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    return (pd.to_datetime(df["time"]).min(), pd.to_datetime(df["time"]).max())

# ---------------- Feature dataset helpers ----------------

def _dataset_for_features(path: str):
    low = str(path).lower()
    if low.endswith((".parquet",".parq",".pq")):
        return "parquet", ds.dataset(path, format="parquet")
    return "csv", None  # CSV handled via pandas chunker

def _csv_batches(path: str, chunksize: int, t0: pd.Timestamp, t1: pd.Timestamp):
    it = pd.read_csv(path, chunksize=chunksize, low_memory=False, parse_dates=["time"])
    for chunk in it:
        mask = (chunk["time"] >= t0) & (chunk["time"] <= t1)
        sub = chunk.loc[mask].copy()
        if not sub.empty:
            yield sub

def _append_csv(path: str, df: pd.DataFrame, first: bool):
    comp = "gzip" if str(path).lower().endswith(".gz") else "infer"
    df.to_csv(path, index=False, mode=("w" if first else "a"), header=first,
              compression=comp, date_format="%Y-%m-%d %H:%M:%S")

# ---------- nearest-neighbor utilities ----------

def _gc_dist_deg(lat1, lon1, lat2, lon2):
    rlat1 = np.radians(lat1); rlat2 = np.radians(lat2)
    rdlon = np.radians(lon1 - lon2)
    rdlon = (rdlon + np.pi) % (2*np.pi) - np.pi
    sin2 = np.sin((rlat2-rlat1)/2.0)**2 + np.cos(rlat1)*np.cos(rlat2)*np.sin(rdlon/2.0)**2
    ang = 2*np.arcsin(np.minimum(1.0, np.sqrt(sin2)))
    return np.degrees(ang)

def _nearest_join_hour(hrows: pd.DataFrame, era_hour: pd.DataFrame, maxdeg: float) -> pd.DataFrame:
    if era_hour.empty or hrows.empty:
        return hrows

    thermo_cols = [c for c in era_hour.columns if c not in ("time","lat","lon")]

    if not HAVE_KDTREE:
        out_vals = {c: np.full(len(hrows), np.nan, dtype=era_hour[c].dtype if c in era_hour else float)
                    for c in thermo_cols}
        for i, (la, lo) in enumerate(hrows[["lat","lon"]].to_numpy()):
            box = era_hour[(np.abs(era_hour["lat"]-la)<=maxdeg) & (np.abs(era_hour["lon"]-lo)<=maxdeg)]
            if box.empty:
                continue
            d = _gc_dist_deg(la, lo, box["lat"].to_numpy(), box["lon"].to_numpy())
            j = int(np.argmin(d))
            if d[j] <= maxdeg:
                row = box.iloc[j]
                for c in thermo_cols: out_vals[c][i] = row[c]
        for c, arr in out_vals.items(): hrows[c] = arr
        return hrows

    lat0 = float(np.clip(np.nanmean(era_hour["lat"].to_numpy()), -80, 80))
    x = era_hour["lon"].to_numpy() * np.cos(np.radians(lat0))
    y = era_hour["lat"].to_numpy()
    tree = cKDTree(np.c_[x, y])

    tx = hrows["lon"].to_numpy() * np.cos(np.radians(lat0))
    ty = hrows["lat"].to_numpy()
    _, idx = tree.query(np.c_[tx, ty], k=1, workers=-1)

    cand = era_hour.iloc[idx].reset_index(drop=True)
    ang = _gc_dist_deg(hrows["lat"].to_numpy(), hrows["lon"].to_numpy(),
                       cand["lat"].to_numpy(), cand["lon"].to_numpy())
    ok = ang <= maxdeg
    for c in thermo_cols:
        vals = cand[c].to_numpy()
        out = np.full(len(hrows), np.nan, dtype=vals.dtype)
        out[ok] = vals[ok]
        hrows[c] = out
    return hrows

# ---------------- main integrate ----------------

def integrate(features_path: str,
              nc_files: List[str],
              out_path: str,
              norm_lon: str,
              vars_wanted: List[str],
              nearest: bool,
              nearest_maxdeg: float,
              row_group_rows: int,
              scan_batch_rows: int,
              csv_chunk_rows: int,
              engine_hint: Optional[str],
              quiet: bool):
    if not nc_files:
        raise SystemExit("No thermo NetCDF files found.")

    feats_kind, feats_ds = _dataset_for_features(features_path)
    feats_is_parquet = (feats_kind == "parquet")

    # Output prep (lazy writer creation to avoid schema issues)
    out_lower = str(out_path).lower()
    is_parquet_out = out_lower.endswith((".parquet",".parq",".pq"))
    outp = Path(out_path)
    if outp.exists():
        outp.unlink()

    writer = None    # pq.ParquetWriter, created on first batch
    csv_first = not is_parquet_out
    total_rows = 0

    # Determine desired column order from a tiny features sample (names only).
    if feats_is_parquet:
        scanner0 = ds.Scanner.from_dataset(feats_ds, columns=None, filter=None, batch_size=min(50_000, scan_batch_rows))
        b0 = next(scanner0.to_batches(), None)
        if b0 is None:
            raise SystemExit("Features table appears empty.")
        feat_cols = list(b0.schema.names)
    else:
        sm = pd.read_csv(features_path, nrows=1000, low_memory=False)
        feat_cols = list(sm.columns)
        del sm

    thermo_cols = [c for c in vars_wanted if c not in ("time","lat","lon")]
    desired_order = feat_cols + [c for c in thermo_cols if c not in feat_cols]

    # Process each monthly NetCDF
    for i, nc in enumerate(sorted(nc_files), 1):
        era = _nc_to_frame(nc, vars_wanted=vars_wanted, norm_lon=norm_lon)
        if not quiet:
            kept = [c for c in era.columns if c not in ("time","lat","lon")]
            print(f"[{i}/{len(nc_files)}] {Path(nc).name}: rows={len(era):,} vars={kept}")
        if era.empty:
            continue

        t0, t1 = _time_bounds(era)

        if feats_is_parquet:
            t0_ns = pd.Timestamp(t0).to_datetime64()
            t1_ns = pd.Timestamp(t1).to_datetime64()
            tf = (ds.field("time") >= pa.scalar(t0_ns, type=pa.timestamp('ns'))) & \
                 (ds.field("time") <= pa.scalar(t1_ns, type=pa.timestamp('ns')))
            scanner = ds.Scanner.from_dataset(feats_ds, columns=None, filter=tf, batch_size=scan_batch_rows)
            batch_iter = scanner.to_batches()
        else:
            batch_iter = _csv_batches(features_path, chunksize=scan_batch_rows, t0=t0, t1=t1)

        # Iterate feature batches
        for batch in batch_iter:
            if feats_is_parquet:
                # Robust: ignore metadata, let pandas infer; downcast later
                pdf = batch.to_pandas(ignore_metadata=True)
            else:
                pdf = batch  # already pandas DataFrame

            if pdf.empty:
                continue

            if not np.issubdtype(pdf["time"].dtype, np.datetime64):
                pdf["time"] = pd.to_datetime(pdf["time"], utc=True, errors="coerce").dt.tz_localize(None)

            # Ensure thermo columns exist
            for c in thermo_cols:
                if c not in pdf.columns:
                    pdf[c] = np.nan

            if not nearest:
                joined = pdf.merge(era, on=["time","lat","lon"], how="left", suffixes=("", "_thermo"))
            else:
                # per-hour nearest map
                for t, idx in pdf.groupby(pdf["time"].dt.floor("H"), sort=False).indices.items():
                    hrows = pdf.loc[idx, ["time","lat","lon"]].copy()
                    era_h = era.loc[era["time"] == t, :]
                    if era_h.empty:
                        continue
                    out_h = _nearest_join_hour(hrows, era_h, maxdeg=nearest_maxdeg)
                    for c in [col for col in era.columns if col not in ("time","lat","lon")]:
                        pdf.loc[idx, c] = out_h[c].to_numpy()
                joined = pdf

            # Downcast float64 → float32 to keep memory in check
            for c in joined.select_dtypes(include=["float64"]).columns:
                joined[c] = joined[c].astype("float32")

            # Reorder columns to stable order (features first, then thermo)
            for c in desired_order:
                if c not in joined.columns:
                    joined[c] = np.nan
            joined = joined[[c for c in desired_order if c in joined.columns]]

            if is_parquet_out:
                table = pa.Table.from_pandas(joined, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, schema=table.schema,
                                              compression="zstd", use_dictionary=True, write_statistics=True)
                writer.write_table(table, row_group_size=row_group_rows)
            else:
                if len(joined) > csv_chunk_rows:
                    for start in range(0, len(joined), csv_chunk_rows):
                        _append_csv(out_path, joined.iloc[start:start+csv_chunk_rows], first=csv_first)
                        csv_first = False
                else:
                    _append_csv(out_path, joined, first=csv_first)
                    csv_first = False

            total_rows += len(joined)
            del joined, pdf
        del era

    if writer is not None:
        writer.close()

    if not quiet:
        print(f"[ok] wrote {out_path}  rows≈{total_rows:,}")

# ---------------- entry ----------------

if __name__ == "__main__":
    args = parse_args()

    thermo_glob = args.thermo_glob or args.nc_glob
    if not thermo_glob:
        print("ERROR: provide --thermo-glob (or --nc-glob).", file=sys.stderr)
        sys.exit(2)

    nc_files = sorted(glob.glob(thermo_glob, recursive=True))
    if not nc_files:
        print("ERROR: No NetCDF files matched.", file=sys.stderr)
        sys.exit(2)

    outp = Path(args.out)
    if outp.exists() and not args.overwrite:
        print(f"ERROR: Outfile exists; use --overwrite: {outp}", file=sys.stderr)
        sys.exit(2)
    if outp.exists() and args.overwrite:
        outp.unlink()

    try:
        integrate(
            features_path=args.features,
            nc_files=nc_files,
            out_path=args.out,
            norm_lon=args.normalize_lon,
            vars_wanted=list(dict.fromkeys(["u10","v10","msl","t2m"] + (args.vars or []))),
            nearest=args.nearest,
            nearest_maxdeg=float(args.nearest_maxdeg),
            row_group_rows=int(args.row_group_rows),
            scan_batch_rows=int(args.scan_batch_rows),
            csv_chunk_rows=int(args.csv_chunk_rows),
            engine_hint=args.engine,
            quiet=args.quiet,
        )
    except Exception as e:
        print(f"ERROR: {type(e).__name__}({e})", file=sys.stderr)
        sys.exit(1)