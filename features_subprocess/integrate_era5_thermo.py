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
* Matching is on (time, lat, lon) using ERA5's own grid coordinates
* Streaming, chunked merge to keep memory bounded; exact (time,lat,lon)
  joins only (no nearest-neighbour yet).

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
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

# Ensure repository root (containing utils/) is importable when run as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import io_common
from utils import join_audit

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
        df = io_common.read_any(
            path,
            compression="infer",
            encoding_errors="replace",
            on_bad_lines="skip",
            parse_dates=["time"],
        )
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    return df


def write_any_table(path: str, df: pd.DataFrame, overwrite: bool = True) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not overwrite:
        print(f"[integrate] exists and overwrite disabled: {p}", flush=True)
        return
    fmt = _detect_table_format(p)
    if fmt == "parquet":
        df.to_parquet(p, index=False)
    else:
        comp = "gzip" if fmt == "csv.gz" else "infer"
        df.to_csv(p, index=False, compression=comp, date_format="%Y-%m-%d %H:%M:%S")


def _detect_table_format(path: Path) -> str:
    """
    Return 'parquet', 'csv', or 'csv.gz' based on suffixes.
    Guard against ambiguous combos like '.parquet.gz' that would silently
    trigger CSV writes with a parquet-looking name.
    """
    suffixes = "".join(path.suffixes[-2:]).lower()
    if suffixes in (".parquet.gz", ".parq.gz", ".pq.gz"):
        raise SystemExit(f"[integrate] Ambiguous extension for {path}; use .parquet or .csv.gz explicitly.")
    if suffixes == ".parquet" or path.suffix.lower() in (".parquet", ".parq", ".pq"):
        return "parquet"
    if suffixes == ".csv.gz":
        return "csv.gz"
    if path.suffix.lower() == ".csv":
        return "csv"
    if path.suffix.lower() == ".gz":
        # treat bare .gz as CSV.GZ to avoid mislabelling parquet payloads
        return "csv.gz"
    raise SystemExit(f"[integrate] Unsupported table format for {path}")


def _iter_feature_chunks(path: Path, chunk_rows: int, parquet_rows: int | None = None):
    """
    Stream the feature table in chunks to keep memory bounded.
    Parquet is streamed via pyarrow batches; CSV/CSV.GZ via pandas chunks.
    """
    fmt = _detect_table_format(path)
    if fmt == "parquet":
        pf = pq.ParquetFile(path)
        batch_size = parquet_rows if parquet_rows and parquet_rows > 0 else chunk_rows
        for batch in pf.iter_batches(batch_size=batch_size):
            yield batch.to_pandas()
        return

    csv_kw = io_common._csv_kwargs(path, {"parse_dates": ["time"], "chunksize": chunk_rows})  # type: ignore[attr-defined]
    # Chunked reads are not supported by pandas' pyarrow engine; fall back to default.
    csv_kw.pop("engine", None)
    csv_kw.pop("dtype_backend", None)
    reader = io_common._read_csv_with_missing_date_guard(path, csv_kw)  # type: ignore[attr-defined]
    chunks = reader if not isinstance(reader, pd.DataFrame) else [reader]
    for chunk in chunks:
        yield chunk


class _ChunkedWriter:
    """
    Minimal streaming writer for CSV(.gz) and Parquet outputs.
    Agent: keep memory usage low by writing chunk-by-chunk instead of buffering
    the whole feature table.
    """

    def __init__(self, path: Path, overwrite: bool):
        self.path = path
        self.format = _detect_table_format(path)
        self._writer = None
        self._wrote_header = False
        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise SystemExit(f"[integrate] exists and overwrite disabled: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, df: pd.DataFrame):
        if self.format == "parquet":
            table = pa.Table.from_pandas(df, preserve_index=False)
            if self._writer is None:
                self._writer = pq.ParquetWriter(self.path, table.schema, compression="snappy")
            self._writer.write_table(table)
        else:
            comp = "gzip" if self.format == "csv.gz" else "infer"
            mode = "w" if not self._wrote_header else "a"
            df.to_csv(
                self.path,
                index=False,
                compression=comp,
                date_format="%Y-%m-%d %H:%M:%S",
                mode=mode,
                header=not self._wrote_header,
            )
            self._wrote_header = True

    def close(self):
        if self._writer is not None:
            self._writer.close()


def _prepare_feat_chunk(df: pd.DataFrame, lon_mode: str) -> pd.DataFrame:
    """Normalize time/lat/lon for a feature chunk and ensure thermo columns exist."""
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce").dt.tz_localize(None)
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce").astype("float32")
    out["lon"] = reframe_lon_vals(pd.to_numeric(out["lon"], errors="coerce").to_numpy(), lon_mode).astype("float32")
    out = out.dropna(subset=["time", "lat", "lon"])

    for col in ("u10", "v10", "msl", "t2m"):
        if col not in out.columns:
            out[col] = np.nan
    return out


def _apply_thermo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Overwrite thermo columns with merged thermo values (if present).
    Existing values are replaced to mirror legacy behaviour.
    """
    for col in ("u10", "v10", "msl", "t2m"):
        thermo_col = f"{col}_thermo"
        if thermo_col in df.columns:
            df[col] = pd.to_numeric(df[thermo_col], errors="coerce")
            df.drop(columns=[thermo_col], inplace=True)
        elif col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _select_thermo_slice(thermo_ranges, tmin, tmax):
    """
    Return thermo DataFrame(s) overlapping [tmin, tmax]. Keeps per-NC slices
    separate to avoid materializing the full thermo table for every chunk.
    """
    relevant = []
    for t0, t1, df in thermo_ranges:
        if (tmin is not None and t0 is not None and t1 is not None and t1 < tmin) or (
            tmax is not None and t0 is not None and t0 > tmax
        ):
            continue
        relevant.append(df)
    if not relevant:
        return None
    if len(relevant) == 1:
        return relevant[0]
    return pd.concat(relevant, ignore_index=True)


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

    # auto-convert MSL Pa->hPa if needed
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
    ap.add_argument("--chunk-rows", type=int, default=0,
                    help="Rows per chunk when streaming features. Default auto-scales with available RAM.")
    ap.add_argument("--chunksize", type=int, default=0,
                    help="Alias for --chunk-rows (kept for orchestrator compatibility).")
    ap.add_argument("--parquet-rows", type=int, default=0,
                    help="Preferred batch size when streaming parquet input/output.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace existing output instead of skipping.")
    # kept for compatibility; we currently do exact (time,lat,lon) join
    ap.add_argument("--nearest", action="store_true",
                    help="(Currently a no-op: exact (time,lat,lon) merge is used.)")
    ap.add_argument("--nearest-maxdeg", type=float, default=0.4,
                    help="Unused placeholder for future nearest-neighbour matching.")
    ap.add_argument("--join-audit-out", default="results/diagnostics/join_audit.json",
                    help="Optional JSON path for join audit logging.")
    ap.add_argument("--allow-many-to-many", action="store_true",
                    help="Allow many-to-many joins without failing.")
    ap.add_argument("--allow-row-explosion", action="store_true",
                    help="Allow output row growth beyond input chunks.")
    return ap.parse_args()


def main():
    args = parse_args()
    feat_path = Path(args.features)
    out_path = Path(args.out)

    if feat_path.resolve() == out_path.resolve():
        raise SystemExit("[integrate] Input and output paths must differ to avoid corruption.")

    thermo_glob = args.thermo_glob or args.nc_glob
    if not thermo_glob:
        raise SystemExit("Must supply --thermo-glob or --nc-glob")

    nc_files = sorted(glob.glob(thermo_glob, recursive=True))
    if not nc_files:
        raise SystemExit(f"No ERA5 thermo files found for pattern: {thermo_glob}")

    area_box = parse_area(args.area)
    lon_mode = _canon_norm(args.normalize_lon)
    in_fmt = _detect_table_format(feat_path)
    out_fmt = _detect_table_format(out_path)
    csv_rows, parq_rows = io_common.recommend_chunk_rows()  # type: ignore[attr-defined]
    chunk_pref = args.chunk_rows or args.chunksize
    parquet_pref = args.parquet_rows
    chunk_rows = chunk_pref or (parq_rows if in_fmt == "parquet" else csv_rows)
    if chunk_rows <= 0:
        chunk_rows = parq_rows if in_fmt == "parquet" else csv_rows
    parquet_rows = parquet_pref or chunk_rows

    print(f"[integrate] features: {feat_path}")
    print(f"[integrate] thermo glob: {thermo_glob} -> {len(nc_files)} files")
    print(f"[integrate] lon mode: {lon_mode}  AOI: {area_box}", flush=True)
    print(f"[integrate] feature format: {in_fmt}  out format: {out_fmt}  chunk_rows: {chunk_rows:,}")
    if args.nearest:
        print("[integrate] --nearest requested; using exact (time,lat,lon) join for now.", flush=True)

    thermo_ranges = []
    thermo_rows_total = 0
    for i, nc in enumerate(nc_files, start=1):
        thermo_df = flatten_era5_single(nc, lon_mode, area_box, engine=args.engine)
        if thermo_df.empty:
            print(f"[integrate] {Path(nc).name}: no thermo rows after AOI; skipping", flush=True)
            continue
        tmin = thermo_df["time"].min()
        tmax = thermo_df["time"].max()
        thermo_ranges.append((tmin, tmax, thermo_df))
        thermo_rows_total += len(thermo_df)
        print(f"[integrate] {i}/{len(nc_files)} {Path(nc).name}: thermo rows {len(thermo_df):,} [{tmin}, {tmax}]", flush=True)

    if not thermo_ranges:
        raise SystemExit("[integrate] No thermo data to merge; aborting.")

    writer = _ChunkedWriter(out_path, overwrite=args.overwrite)
    col_order = None
    rows_written = 0
    # Agent: join audit counters (no math changes).
    audit_left_rows = 0
    audit_out_rows = 0
    audit_left_dupe = 0
    audit_right_dupe_max = 0
    audit_row_explode_chunks = 0
    audit_chunks = 0

    for i, raw_chunk in enumerate(_iter_feature_chunks(feat_path, chunk_rows, parquet_rows=parquet_rows), start=1):
        if raw_chunk is None or len(raw_chunk) == 0:
            continue
        if not {"time", "lat", "lon"}.issubset(raw_chunk.columns):
            raise SystemExit("Feature table must include columns: time, lat, lon")

        chunk = _prepare_feat_chunk(raw_chunk, lon_mode)
        if chunk.empty:
            continue

        tmin = chunk["time"].min()
        tmax = chunk["time"].max()
        thermo_slice = _select_thermo_slice(thermo_ranges, tmin, tmax)
        if thermo_slice is not None:
            merged = chunk.merge(
                thermo_slice,
                on=["time", "lat", "lon"],
                how="left",
                suffixes=("", "_thermo"),
            )
        else:
            merged = chunk.copy()
        merged = _apply_thermo_columns(merged)

        # Join audit per-chunk (fail fast on explosions).
        left_dupe = int(chunk.duplicated(subset=["time", "lat", "lon"]).sum())
        right_dupe = int(thermo_slice.duplicated(subset=["time", "lat", "lon"]).sum()) if thermo_slice is not None else 0
        chunk_entry = join_audit.build_entry(
            step="features.integrate-thermo",
            keys=["time", "lat", "lon"],
            join_type="left",
            left_rows=len(chunk),
            right_rows=len(thermo_slice) if thermo_slice is not None else 0,
            out_rows=len(merged),
            left_dupe_keys=left_dupe,
            right_dupe_keys=right_dupe,
            extra={"chunk": i},
        )
        join_audit.enforce(
            chunk_entry,
            allow_many_to_many=args.allow_many_to_many,
            allow_row_explosion=args.allow_row_explosion,
        )
        if chunk_entry.get("row_explosion"):
            audit_row_explode_chunks += 1
        audit_left_dupe += left_dupe
        audit_right_dupe_max = max(audit_right_dupe_max, right_dupe)
        audit_left_rows += len(chunk)
        audit_out_rows += len(merged)
        audit_chunks += 1

        if col_order is None:
            col_order = list(merged.columns)
        else:
            missing = [c for c in col_order if c not in merged.columns]
            for c in missing:
                merged[c] = np.nan
            extra = [c for c in merged.columns if c not in col_order]
            if extra:
                col_order.extend(extra)
        merged = merged[col_order]

        writer.write(merged)
        rows_written += len(merged)
        if i % 5 == 0:
            print(f"[integrate] chunk {i}: rows={len(merged):,} written_total={rows_written:,}", flush=True)
        del raw_chunk, chunk, merged, thermo_slice

    writer.close()
    audit_entry = join_audit.build_entry(
        step="features.integrate-thermo",
        keys=["time", "lat", "lon"],
        join_type="left",
        left_rows=audit_left_rows,
        right_rows=thermo_rows_total,
        out_rows=audit_out_rows,
        left_dupe_keys=audit_left_dupe,
        right_dupe_keys=audit_right_dupe_max,
        extra={
            "chunks": audit_chunks,
            "row_explosion_chunks": audit_row_explode_chunks,
            "features_path": str(feat_path),
            "out_path": str(out_path),
        },
    )
    join_audit.append_entry(args.join_audit_out, audit_entry)
    print(f"[integrate] wrote {out_path} rows={rows_written:,} thermo_rows={thermo_rows_total:,}", flush=True)


if __name__ == "__main__":
    main()
