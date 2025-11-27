#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage_data.py — fetch/clean/stage inputs and emit a manifest (now with embedded config).

What it does (per file):
  • CSV/CSV.GZ only: normalize timestamps to tz-naive UTC (YYYY-mm-dd HH:MM:SS)
  • Optional: normalize longitudes to -180..180 or 0..360
  • Optional: crop to an AOI N,W,S,E
  • Writes back IN-PLACE safely via a temporary file swap
  • Parquet: never modified; only scanned for manifest stats

Extras:
  • Optional ID/key-table + slim-scoring table from a rich CSV/Parquet:
      - Compute row_id from (time,lat,lon) via 64-bit hash
      - Write key table: row_id,time,lat,lon
      - Optionally write full table: row_id + all original columns
      - Write slim table: row_id + selected feature columns

Config precedence:
  1) CLI flags (highest)
  2) External --config YAML/JSON, if provided
  3) Embedded config (when --use-embedded-config is set)
  4) Built-in defaults (lowest)

Handy extras:
  • --print-embedded-config
  • --save-embedded-config path
"""

from __future__ import annotations
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

# ===================== Embedded Config (edit this block) =====================
# Keep values realistic but safe; empty lists mean "no-op" until you fill them.
EMBEDDED_CONFIG: Dict[str, Any] = {
    # One or more globs of CSV/CSV.GZ files to normalize in-place:
    "sanitize": [],
    # Extra files/globs to include in the manifest without rewriting:
    "scan": [],
    # Column with timestamps (if present):
    "time_col": "time",
    # Optional strptime format (e.g., "%Y-%m-%d %H:%M:%S"):
    "time_format": None,
    # "-180..180", "0..360", or "none"
    "normalize_lon": "-180..180",
    # Optional AOI "latN,lonW,latS,lonE" as a single string (or None):
    # Example for AU region box: "-5,125,-35,175"
    "area": None,
    # CSV chunk size
    "chunk_rows": 1_000_000,
    # Where to write the manifest JSON
    "manifest": "staged/manifest.json",
    # Base path for relative manifest paths
    "manifest_root": ".",
    # Compute SHA256 for each file (slower)
    "hash": False,

    # -------- Optional: ID + slim scoring tables from rich CSV/Parquet --------
    # No-ops unless id_build_enable is true.
    #
    # Typical setup for your use-case:
    #   id_build_enable: true
    #   id_src: data/grid_labelled_FMA_gka_realthermo.parquet
    #   id_extra_src: data/grid_labelled_FMA_gka_realthermo_sph.csv.gz
    #   id_key_out: data/grid_key_table.csv.gz
    #   id_full_out: data/grid_labelled_FMA_gka_realthermo_with_id.csv.gz
    #   id_slim_out: data/grid_scoring_start.csv.gz
    #   id_slim_cols: ["wspd","shear_low","S3","msl","u10","t2m","zeta",
    #                  "shear_deep","S","div","v10","SFI","SFI2",...]
    "id_build_enable": False,
    "id_src": None,
    "id_extra_src": None,   # optional extra file to merge on time/lat/lon
    "id_key_out": None,
    "id_full_out": None,
    "id_slim_out": None,
    "id_slim_cols": [],
    "id_chunk_rows": None,  # defaults to chunk_rows when None
}
# ============================================================================

# -------------------- helpers --------------------

def _is_parquet(p: Path) -> bool:
    return p.suffix.lower() in {".parquet", ".parq", ".pq", ".pqt"}


def _is_csv_like(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".csv") or n.endswith(".csv.gz") or n.endswith(".gz")


def _sha256_file(path: Path, bufsize: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_utc_naive(series: pd.Series, fmt: Optional[str]) -> pd.Series:
    # accept already-datetime; otherwise parse
    if pd.api.types.is_datetime64_any_dtype(series):
        t = pd.to_datetime(series, utc=True, errors="coerce")
        return t.dt.tz_convert(None)
    raw = series.astype(str).str.strip().str.replace("Z", "", regex=False)
    t = pd.to_datetime(raw, format=fmt, utc=True, errors="coerce") if fmt else pd.to_datetime(
        raw, utc=True, errors="coerce"
    )
    return t.dt.tz_convert(None)


def _canon_norm_mode(mode: str) -> str:
    """Accepts ' -180..180' and other spacey variants."""
    if mode is None:
        return "-180..180"
    t = str(mode).strip().replace(" ", "")
    if t in {"-180..180", "0..360"}:
        return t
    if t.lower() == "none":
        return "none"
    return "-180..180"


def _norm_lon(vals: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(vals, errors="coerce")
    mode = _canon_norm_mode(mode)
    if mode == "none":
        return x
    if mode == "0..360":
        return (x % 360 + 360) % 360
    # default → "-180..180"
    return ((x + 180.0) % 360.0) - 180.0


def _parse_area(aoi: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not aoi:
        return None
    latN, lonW, latS, lonE = [float(z.strip()) for z in str(aoi).split(",")]
    return latN, lonW, latS, lonE


def _apply_aoi(df: pd.DataFrame, aoi: Tuple[float, float, float, float], latc: str, lonc: str) -> pd.DataFrame:
    N, W, S, E = aoi
    latv = pd.to_numeric(df[latc], errors="coerce")
    lonv = pd.to_numeric(df[lonc], errors="coerce")
    return df.loc[(latv <= N) & (latv >= S) & (lonv >= W) & (lonv <= E)]


def _first_present(cols: List[str], cands: List[str]) -> Optional[str]:
    s = set(cols)
    for c in cands:
        if c in s:
            return c
    return None


def _lat_lon_names(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    lat = _first_present(list(df.columns), ["lat", "latitude", "Lat", "Latitude"])
    lon = _first_present(list(df.columns), ["lon", "longitude", "Lon", "Longitude"])
    return lat, lon


def _safe_swap_write(tmp_path: Path, dest_path: Path):
    # Windows-safe replace
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        dest_path.unlink()
    tmp_path.replace(dest_path)


def _print(*a, **k):
    print(*a, **k, flush=True)


# -------------------- config loader/merger --------------------

def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(y) for y in x]
    return [str(x)]


def load_config_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--config file not found: {p}")
    text = p.read_text(encoding="utf-8")
    # Try YAML first, fall back to JSON
    try:
        import yaml  # type: ignore
        cfg = yaml.safe_load(text)
        if cfg is None:
            return {}
    except Exception:
        try:
            cfg = json.loads(text)
        except Exception as e:
            raise ValueError(f"Unable to parse config as YAML or JSON: {e}")
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping/object.")
    # Normalize keys (dash/underscore)
    out: Dict[str, Any] = {}
    for k, v in cfg.items():
        out[k.replace("-", "_")] = v
    return out


def normalize_config_shapes(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    out["sanitize"] = _as_list(out.get("sanitize"))
    out["scan"] = _as_list(out.get("scan"))
    for key in (
        "time_col",
        "time_format",
        "normalize_lon",
        "area",
        "manifest",
        "manifest_root",
        "id_src",
        "id_extra_src",
        "id_key_out",
        "id_full_out",
        "id_slim_out",
    ):
        if key in out and out[key] is not None:
            out[key] = str(out[key])
    if "chunk_rows" in out and out["chunk_rows"] is not None:
        out["chunk_rows"] = int(out["chunk_rows"])
    if "hash" in out:
        out["hash"] = bool(out["hash"])
    # ID builder bits
    if "id_build_enable" in out:
        out["id_build_enable"] = bool(out["id_build_enable"])
    # IMPORTANT: expand id_slim_cols including comma-split, so
    # "--id-slim-cols SFI,SFI2,..." becomes ["SFI","SFI2",...]
    raw_cols = out.get("id_slim_cols")
    if raw_cols is None:
        out["id_slim_cols"] = []
    else:
        pieces: List[str] = []
        for item in _as_list(raw_cols):
            for part in str(item).split(","):
                part = part.strip()
                if part:
                    pieces.append(part)
        out["id_slim_cols"] = pieces
    if "id_chunk_rows" in out and out["id_chunk_rows"] is not None:
        out["id_chunk_rows"] = int(out["id_chunk_rows"])
    return out


def merge_cli_over(base: Dict[str, Any], override_ns: argparse.Namespace) -> Dict[str, Any]:
    defaults = {
        "sanitize": [],
        "scan": [],
        "time_col": "time",
        "time_format": None,
        "normalize_lon": "-180..180",
        "area": None,
        "chunk_rows": 1_000_000,
        "manifest": "staged/manifest.json",
        "manifest_root": ".",
        "hash": False,
        # ID builder defaults
        "id_build_enable": False,
        "id_src": None,
        "id_extra_src": None,
        "id_key_out": None,
        "id_full_out": None,
        "id_slim_out": None,
        "id_slim_cols": [],
        "id_chunk_rows": None,
    }
    merged = dict(defaults)
    merged.update(base or {})
    # merge CLI on top where provided
    for k in defaults.keys():
        cli_val = getattr(override_ns, k, defaults[k])
        if cli_val is not None and cli_val != defaults[k] and not (
            isinstance(cli_val, list) and cli_val == []
        ):
            merged[k] = cli_val
    return merged


# -------------------- core: sanitize CSV in place --------------------

def sanitize_csv_inplace(
    path: Path,
    time_col: str,
    time_fmt: Optional[str],
    normalize_lon: str,
    area: Optional[Tuple[float, float, float, float]],
    chunk_rows: int = 1_000_000,
) -> Dict:
    """
    Read a CSV(.gz) in chunks, normalize time/lon, optional AOI crop, and rewrite file in place.
    Returns basic stats for manifest.
    """
    low = path.name.lower()
    compression = "gzip" if low.endswith(".gz") else "infer"
    tmp = Path(tempfile.gettempdir()) / f"__staging_{path.name}.tmp"

    wrote_header = False
    rows_out = 0
    cols_seen: Optional[int] = None
    time_min = None
    time_max = None
    lat_min = lon_min = np.inf
    lat_max = lon_max = -np.inf

    # stream read
    rdr = pd.read_csv(path, compression=compression, low_memory=False, chunksize=chunk_rows)
    if not hasattr(rdr, "__iter__"):
        rdr = [rdr]

    # ensure empty tmp file
    with tmp.open("wb") as _:
        pass

    for i, chunk in enumerate(rdr, start=1):
        if chunk is None or chunk.empty:
            continue

        # fix BOM on time column if needed
        if time_col not in chunk.columns:
            bom = "\ufeff" + time_col
            if bom in chunk.columns:
                chunk = chunk.rename(columns={bom: time_col})

        if time_col in chunk.columns:
            t = _to_utc_naive(chunk[time_col], time_fmt)
            good = t.notna()
            if good.any():
                chunk = chunk.loc[good].copy()
                chunk[time_col] = t[good]

        # optional lon normalize / AOI crop
        latc, lonc = _lat_lon_names(chunk)
        if lonc is not None:
            chunk[lonc] = _norm_lon(chunk[lonc], normalize_lon)
        if area is not None and (latc is not None) and (lonc is not None):
            chunk = _apply_aoi(chunk, area, latc, lonc)

        if chunk.empty:
            continue

        # gather stats
        if cols_seen is None:
            cols_seen = len(chunk.columns)
        rows_out += len(chunk)
        if time_col in chunk.columns:
            tmin = chunk[time_col].min()
            tmax = chunk[time_col].max()
            time_min = tmin if time_min is None else min(time_min, tmin)
            time_max = tmax if time_max is None else max(time_max, tmax)
        if latc and lonc:
            latv = pd.to_numeric(chunk[latc], errors="coerce")
            lonv = pd.to_numeric(chunk[lonc], errors="coerce")
            if latv.notna().any():
                lat_min = float(min(lat_min, latv.min()))
                lat_max = float(max(lat_max, latv.max()))
            if lonv.notna().any():
                lon_min = float(min(lon_min, lonv.min()))
                lon_max = float(max(lon_max, lonv.max()))

        # append write
        chunk.to_csv(
            tmp,
            index=False,
            mode="a",
            header=not wrote_header,
            compression=compression,
            date_format="%Y-%m-%d %H:%M:%S",
        )
        wrote_header = True
        _print(f"[stage] {path.name}: chunk {i} → {len(chunk)} rows")

    # swap into place if we wrote anything; if empty, keep original but report zero effective rows
    if wrote_header:
        _safe_swap_write(tmp, path)
        _print(f"[stage] wrote → {path}   rows={rows_out}")
    else:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        _print(f"[stage] no rows after filters: left original file intact → {path}")

    # stats
    if not np.isfinite(lat_min):  # never saw lat/lon
        lat_min = lat_max = lon_min = lon_max = np.nan

    return dict(
        rows=rows_out,
        cols=(cols_seen or 0),
        time_min=str(time_min) if time_min is not None else "",
        time_max=str(time_max) if time_max is not None else "",
        lat_min=None if np.isnan(lat_min) else float(lat_min),
        lat_max=None if np.isnan(lat_max) else float(lat_max),
        lon_min=None if np.isnan(lon_min) else float(lon_min),
        lon_max=None if np.isnan(lon_max) else float(lon_max),
    )


# -------------------- generic chunk iterator for CSV / Parquet --------------------

def _iter_table_chunks(src: Path, chunk_rows: int = 500_000):
    """
    Yield pandas DataFrame chunks from a CSV(.gz) or Parquet file
    without loading the whole thing into memory.
    """
    if _is_parquet(src):
        # Prefer streaming by row-group via pyarrow; fall back to a single pandas.read_parquet
        try:
            import pyarrow.parquet as pq  # type: ignore

            pf = pq.ParquetFile(str(src))
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg)
                df = table.to_pandas()
                if df is not None and not df.empty:
                    yield df
        except Exception as e:
            _print(f"[id-build] parquet streaming failed ({e}); falling back to pandas.read_parquet")
            df = pd.read_parquet(src)
            if df is not None and not df.empty:
                yield df
    else:
        low = src.name.lower()
        compression = "gzip" if low.endswith(".gz") else "infer"
        rdr = pd.read_csv(src, compression=compression, low_memory=False, chunksize=chunk_rows)
        if not hasattr(rdr, "__iter__"):
            rdr = [rdr]
        for ch in rdr:
            if ch is not None and not ch.empty:
                yield ch


# -------------------- ID builder helpers --------------------

def _guess_time_col(cols: List[str], hint: str = "time") -> Optional[str]:
    if hint in cols:
        return hint
    return _first_present(cols, ["time", "datetime", "valid_time", "time_h"])


def build_ids_from_csv(
    src: Path,
    key_out: Path,
    full_out: Optional[Path],
    slim_out: Optional[Path],
    slim_cols: List[str],
    chunk_rows: int = 500_000,
    time_col_hint: str = "time",
    extra_src: Optional[Path] = None,
) -> Dict:
    """
    Stream a rich CSV(.gz) or Parquet file and build:
      • key table:  row_id, time, lat, lon      (time floored to hour)
      • full table: row_id + all original cols  (optional)
      • slim table: row_id + selected features  (optional)

    row_id is a 64-bit stable hash of (time_floor_H, lat_round3, lon_round3),
    via pandas.util.hash_pandas_object.

    If extra_src is provided, non-key columns from extra_src are merged into
    each chunk by (time, lat, lon). Any duplicates in extra_src for a given
    key are collapsed by keeping the first row for that key in the chunk.
    """

    # --- normalize slim_cols so we tolerate comma-joined args ---
    norm_cols: List[str] = []
    for item in (slim_cols or []):
        for part in str(item).split(","):
            part = part.strip()
            if part:
                norm_cols.append(part)
    slim_cols = norm_cols
    # ------------------------------------------------------------

    key_out.parent.mkdir(parents=True, exist_ok=True)
    if full_out is not None:
        full_out.parent.mkdir(parents=True, exist_ok=True)
    if slim_out is not None:
        slim_out.parent.mkdir(parents=True, exist_ok=True)

    key_first = True
    full_first = True
    slim_first = True

    rows_out = 0
    cols_seen: Optional[int] = None
    time_min = None
    time_max = None
    lat_min = lon_min = np.inf
    lat_max = lon_max = -np.inf

    _print(
        f"[id-build] src={src} extra={extra_src or '(none)'} → key={key_out} "
        f"full={full_out or '(none)'} slim={slim_out or '(none)'} chunk_rows={chunk_rows}"
    )

    key_comp = "gzip" if key_out.name.lower().endswith(".gz") else "infer"
    full_comp = "gzip" if (full_out and full_out.name.lower().endswith(".gz")) else "infer"
    slim_comp = "gzip" if (slim_out and slim_out.name.lower().endswith(".gz")) else "infer"

    extra_iter = None
    if extra_src is not None and extra_src.exists():
        _print(f"[id-build] enabling extra merge from {extra_src}")
        extra_iter = _iter_table_chunks(extra_src, chunk_rows)
    elif extra_src is not None:
        _print(f"[id-build] extra_src specified but not found: {extra_src}; skipping extra merge.")

    for i, base_chunk in enumerate(_iter_table_chunks(src, chunk_rows), start=1):
        chunk = base_chunk

        cols = list(chunk.columns)
        if cols_seen is None:
            cols_seen = len(cols)
            _print(f"[id-build] first chunk columns: {cols}")
            _print(f"[id-build] requested slim cols: {slim_cols}")

        tcol = _guess_time_col(cols, hint=time_col_hint)
        latc, lonc = _lat_lon_names(chunk)

        if not tcol or not latc or not lonc:
            _print(f"[id-build] {src.name}: missing time/lat/lon in chunk {i}, skipping.")
            continue

        # Normalized keys for base
        base_time = _to_utc_naive(chunk[tcol], None)
        base_lat = pd.to_numeric(chunk[latc], errors="coerce").round(3)
        base_lon = pd.to_numeric(chunk[lonc], errors="coerce").round(3)
        base_keys = pd.DataFrame({"time": base_time, "lat": base_lat, "lon": base_lon})

        # --- attempt to merge extra_src on (time, lat, lon) for this chunk ---
        if extra_iter is not None:
            try:
                extra_chunk = next(extra_iter)
            except StopIteration:
                _print("[id-build] extra_src exhausted; no further extra merges.")
                extra_iter = None
                extra_chunk = None

            if extra_chunk is not None:
                extra_cols_list = list(extra_chunk.columns)
                tcol_extra = _guess_time_col(extra_cols_list, hint=time_col_hint)
                latc_extra, lonc_extra = _lat_lon_names(extra_chunk)

                if not tcol_extra or not latc_extra or not lonc_extra:
                    _print(
                        f"[id-build] chunk {i}: extra_src missing time/lat/lon; "
                        "disabling extra merge for remaining chunks."
                    )
                    extra_iter = None
                else:
                    extra_time = _to_utc_naive(extra_chunk[tcol_extra], None)
                    extra_lat = pd.to_numeric(extra_chunk[latc_extra], errors="coerce").round(3)
                    extra_lon = pd.to_numeric(extra_chunk[lonc_extra], errors="coerce").round(3)

                    extra_nonkey = [
                        c for c in extra_cols_list
                        if c not in {tcol_extra, latc_extra, lonc_extra}
                    ]

                    if extra_nonkey:
                        extra_join = pd.concat(
                            [
                                pd.DataFrame({"time": extra_time, "lat": extra_lat, "lon": extra_lon}),
                                extra_chunk[extra_nonkey].reset_index(drop=True),
                            ],
                            axis=1,
                        )

                        # collapse duplicates per (time,lat,lon) in extra_src chunk
                        before = len(extra_join)
                        extra_join = (
                            extra_join
                            .sort_values(["time", "lat", "lon"])
                            .drop_duplicates(subset=["time", "lat", "lon"], keep="first")
                        )
                        dup_dropped = before - len(extra_join)
                        if dup_dropped > 0:
                            _print(
                                f"[id-build] chunk {i}: dropped {dup_dropped} duplicate "
                                "extra rows on (time,lat,lon)."
                            )

                        merged = base_keys.merge(
                            extra_join,
                            on=["time", "lat", "lon"],
                            how="left",
                            sort=False,
                        )

                        # sanity: must remain one row per base row
                        if len(merged) != len(chunk):
                            _print(
                                f"[id-build] chunk {i}: merged rows={len(merged)} != base rows={len(chunk)}; "
                                "disabling extra merge for remaining chunks."
                            )
                            extra_iter = None
                        else:
                            for col in extra_nonkey:
                                if col in chunk.columns:
                                    continue  # keep base version
                                chunk[col] = merged[col].values

                            _print(
                                f"[id-build] chunk {i}: merged extra cols {extra_nonkey} "
                                f"(matches={merged[extra_nonkey].notna().any(axis=1).sum():,})"
                            )
                    else:
                        _print(f"[id-build] chunk {i}: no extra non-key columns; nothing to merge.")

        # --- build IDs and outputs from (possibly merged) chunk ---

        # floor time to hour for ID
        t_floor = base_time.dt.floor("H")

        hash_frame = pd.DataFrame({
            "time": t_floor,
            "lat": base_lat,
            "lon": base_lon,
        })
        row_id = hash_pandas_object(hash_frame, index=False).astype("uint64")

        key_chunk = pd.DataFrame({
            "row_id": row_id,
            "time": t_floor,
            "lat": base_lat,
            "lon": base_lon,
        })

        # stats
        rows_out += len(key_chunk)
        if t_floor.notna().any():
            tmin = t_floor.min()
            tmax = t_floor.max()
            time_min = tmin if time_min is None else min(time_min, tmin)
            time_max = tmax if time_max is None else max(time_max, tmax)
        if base_lat.notna().any():
            lat_min = float(min(lat_min, base_lat.min()))
            lat_max = float(max(lat_max, base_lat.max()))
        if base_lon.notna().any():
            lon_min = float(min(lon_min, base_lon.min()))
            lon_max = float(max(lon_max, base_lon.max()))

        # key table
        key_chunk.to_csv(
            key_out,
            index=False,
            mode="w" if key_first else "a",
            header=key_first,
            compression=key_comp,
            date_format="%Y-%m-%d %H:%M:%S",
        )
        key_first = False

        # full table
        if full_out is not None:
            full_chunk = chunk.copy()
            full_chunk.insert(0, "row_id", row_id.values)
            full_chunk.to_csv(
                full_out,
                index=False,
                mode="w" if full_first else "a",
                header=full_first,
                compression=full_comp,
                date_format="%Y-%m-%d %H:%M:%S",
            )
            full_first = False

        # slim scoring table
        if slim_out is not None and slim_cols:
            present = [c for c in slim_cols if c in chunk.columns]
            if present:
                slim_chunk = chunk[present].copy()
                slim_chunk.insert(0, "row_id", row_id.values)
                slim_chunk.to_csv(
                    slim_out,
                    index=False,
                    mode="w" if slim_first else "a",
                    header=slim_first,
                    compression=slim_comp,
                    date_format="%Y-%m-%d %H:%M:%S",
                )
                slim_first = False
            else:
                _print(f"[id-build] chunk {i}: none of requested slim columns present; skipping slim write.")

        _print(f"[id-build] {src.name}: chunk {i} rows={len(chunk)} (total_out={rows_out})")

    if not np.isfinite(lat_min):
        lat_min = lat_max = lon_min = lon_max = np.nan

    return dict(
        rows=int(rows_out),
        cols=int(cols_seen or 0),
        time_min=str(time_min) if time_min is not None else "",
        time_max=str(time_max) if time_max is not None else "",
        lat_min=None if np.isnan(lat_min) else float(lat_min),
        lat_max=None if np.isnan(lat_max) else float(lat_max),
        lon_min=None if np.isnan(lon_min) else float(lon_min),
        lon_max=None if np.isnan(lon_max) else float(lon_max),
    )


# -------------------- scan only (Parquet / read-only) --------------------

def scan_only(path: Path, time_col_hint: str = "time") -> Dict:
    bytes_ = path.stat().st_size if path.exists() else 0
    dtype = "parquet" if _is_parquet(path) else ("csv" if _is_csv_like(path) else "file")
    cols = 0
    rows = 0
    time_min = time_max = ""
    lat_min = lat_max = lon_min = lon_max = None

    try:
        df = None

        if _is_parquet(path):
            # Metadata-only path to avoid loading giant tables into RAM
            try:
                import pyarrow.parquet as pq  # type: ignore

                pf = pq.ParquetFile(str(path))
                md = pf.metadata
                if md is not None:
                    rows = md.num_rows or 0
                    cols = md.num_columns or 0
                # We deliberately skip reading full row-groups; leave time/lat/lon stats empty
            except Exception as e:
                _print(f"[scan] {path.name}: parquet metadata read failed ({e}); falling back to pandas.")
                df = pd.read_parquet(path, columns=None)

        elif _is_csv_like(path):
            # Sample only; we don't need full CSV either
            df = pd.read_csv(path, nrows=1000, compression="infer", low_memory=False)

        if df is not None and not df.empty:
            if cols == 0:
                cols = len(df.columns)
            if _is_parquet(path) and rows == 0:
                try:
                    rows = int(df.shape[0])
                except Exception:
                    rows = 0
            tcol = time_col_hint if time_col_hint in df.columns else _first_present(
                list(df.columns), ["time", "datetime", "valid_time", "time_h"]
            )
            if tcol:
                tt = pd.to_datetime(df[tcol], errors="coerce", utc=True).dt.tz_convert(None)
                if tt.notna().any():
                    time_min = str(tt.min())
                    time_max = str(tt.max())
            latc, lonc = _lat_lon_names(df)
            if latc and lonc:
                lat = pd.to_numeric(df[latc], errors="coerce")
                lon = pd.to_numeric(df[lonc], errors="coerce")
                if lat.notna().any():
                    lat_min, lat_max = float(lat.min()), float(lat.max())
                if lon.notna().any():
                    lon_min, lon_max = float(lon.min()), float(lon.max())
    except Exception as e:
        _print(f"[scan] {path.name}: {e}")

    return dict(
        rows=rows,
        cols=cols,
        time_min=time_min,
        time_max=time_max,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        bytes=bytes_,
        type=dtype,
    )


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser(description="Stage data (sanitize & manifest).")
    ap.add_argument(
        "--config",
        default=None,
        help="YAML/JSON config file. CLI overrides values in this file.",
    )
    ap.add_argument(
        "--use-embedded-config",
        action="store_true",
        help="Apply the embedded config block at the top of this script.",
    )
    ap.add_argument(
        "--print-embedded-config",
        action="store_true",
        help="Print the embedded config (JSON) and exit.",
    )
    ap.add_argument(
        "--save-embedded-config",
        default=None,
        help="Save the embedded config (JSON) to the given path and exit.",
    )

    ap.add_argument(
        "--sanitize",
        nargs="*",
        default=[],
        help="Glob(s) of CSV/CSV.GZ files to clean in place",
    )
    ap.add_argument(
        "--scan",
        nargs="*",
        default=[],
        help="Extra files/globs to include in the manifest (no rewrite)",
    )
    ap.add_argument(
        "--time-col",
        dest="time_col",
        default="time",
        help="Timestamp column (default: time)",
    )
    ap.add_argument(
        "--time-format",
        dest="time_format",
        default=None,
        help="Optional strptime format",
    )
    ap.add_argument(
        "--normalize-lon",
        dest="normalize_lon",
        choices=["-180..180", "0..360", "none"],
        default="-180..180",
    )
    ap.add_argument(
        "--area",
        default=None,
        help='Optional AOI "latN,lonW,latS,lonE"',
    )
    ap.add_argument(
        "--chunk-rows",
        dest="chunk_rows",
        type=int,
        default=1_000_000,
        help="CSV chunk size",
    )
    ap.add_argument(
        "--manifest",
        default="staged/manifest.json",
        help="Manifest JSON path",
    )
    ap.add_argument(
        "--manifest-root",
        dest="manifest_root",
        default=".",
        help="Base path for relative manifest paths",
    )
    ap.add_argument(
        "--hash",
        action="store_true",
        help="Compute SHA256 for each file (slower)",
    )

    # ID builder CLI
    ap.add_argument(
        "--id-build-enable",
        dest="id_build_enable",
        action="store_true",
        help="Enable ID/key+slim generation from a rich CSV/Parquet file",
    )
    ap.add_argument(
        "--id-src",
        dest="id_src",
        default=None,
        help="Source rich CSV(.gz) or Parquet to derive row_id/time/lat/lon from",
    )
    ap.add_argument(
        "--id-extra-src",
        dest="id_extra_src",
        default=None,
        help="Optional extra CSV/Parquet with matching time/lat/lon to merge before ID build",
    )
    ap.add_argument(
        "--id-key-out",
        dest="id_key_out",
        default=None,
        help="Output key CSV(.gz) with row_id,time,lat,lon",
    )
    ap.add_argument(
        "--id-full-out",
        dest="id_full_out",
        default=None,
        help="Optional output full CSV(.gz) with row_id + all original columns (plus extras)",
    )
    ap.add_argument(
        "--id-slim-out",
        dest="id_slim_out",
        default=None,
        help="Optional output slim CSV(.gz) with row_id + selected features",
    )
    ap.add_argument(
        "--id-slim-cols",
        dest="id_slim_cols",
        nargs="*",
        default=None,
        help="Columns to keep in slim scoring table",
    )
    ap.add_argument(
        "--id-chunk-rows",
        dest="id_chunk_rows",
        type=int,
        default=None,
        help="Chunk size for ID builder (defaults to chunk_rows)",
    )

    ns = ap.parse_args()

    if ns.print_embedded_config:
        print(json.dumps(EMBEDDED_CONFIG, indent=2))
        return
    if ns.save_embedded_config:
        Path(ns.save_embedded_config).write_text(
            json.dumps(EMBEDDED_CONFIG, indent=2), encoding="utf-8"
        )
        print(f"Saved embedded config → {ns.save_embedded_config}")
        return

    # Build config base (external, embedded, or empty), then merge CLI on top
    base_cfg: Dict[str, Any] = {}
    if ns.config:
        base_cfg = load_config_file(ns.config)
    elif ns.use_embedded_config:
        base_cfg = dict(EMBEDDED_CONFIG)

    base_cfg = normalize_config_shapes(base_cfg)
    merged = merge_cli_over(base_cfg, ns)

    # Harmonize normalize-lon variants
    merged["normalize_lon"] = _canon_norm_mode(merged.get("normalize_lon"))

    area = _parse_area(merged.get("area"))
    root = Path(merged.get("manifest_root", ".")).resolve()
    out_manifest = Path(merged.get("manifest", "staged/manifest.json"))

    # Expand globs
    def _expand_many(pats: List[str]) -> List[Path]:
        files: List[Path] = []
        for pat in pats:
            for s in glob.glob(pat, recursive=True):
                files.append(Path(s))
        files = sorted({p.resolve() for p in files})  # unique + stable
        return files

    sanitize_targets = _expand_many(merged.get("sanitize", []))
    scan_targets = _expand_many(merged.get("scan", []))

    if (not sanitize_targets and not scan_targets and not merged.get("id_build_enable")):
        _print("[stage] Nothing to do. Provide --sanitize/--scan or enable id-build.")
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        out_manifest.write_text(json.dumps({"files": []}, indent=2), encoding="utf-8")
        return

    _print(f"[stage] sanitize {len(sanitize_targets)} file(s); scan {len(scan_targets)} file(s).")

    records: List[Dict] = []

    # Process sanitize targets
    for p in sanitize_targets:
        if not p.exists():
            _print(f"[stage] missing, skip: {p}")
            continue
        rec_base = dict(
            path=str(p.relative_to(root)) if str(p).startswith(str(root)) else str(p),
            bytes=p.stat().st_size,
            type="csv" if _is_csv_like(p) else ("parquet" if _is_parquet(p) else "file"),
        )
        if _is_csv_like(p):
            stats = sanitize_csv_inplace(
                p,
                time_col=merged["time_col"],
                time_fmt=merged["time_format"],
                normalize_lon=merged["normalize_lon"],
                area=area,
                chunk_rows=int(merged["chunk_rows"]),
            )
        else:
            _print(f"[stage] Not a CSV-like file, leaving as-is: {p.name}")
            stats = scan_only(p, time_col_hint=merged["time_col"])

        rec = {**rec_base, **stats}
        if merged.get("hash", False):
            try:
                rec["sha256"] = _sha256_file(p)
            except Exception as e:
                rec["sha256"] = f"ERROR: {e}"
        records.append(rec)

    # Scan-only targets
    for p in scan_targets:
        if not p.exists():
            _print(f"[scan] missing, skip: {p}")
            continue
        stats = scan_only(p, time_col_hint=merged["time_col"])
        # Avoid duplicate keys when building dict(...)
        stats.pop("bytes", None)
        typ = stats.pop("type", None) or (
            "parquet" if _is_parquet(p) else ("csv" if _is_csv_like(p) else "file")
        )
        rec = dict(
            path=str(p.relative_to(root)) if str(p).startswith(str(root)) else str(p),
            bytes=p.stat().st_size,
            type=typ,
            **stats,
        )
        if merged.get("hash", False):
            try:
                rec["sha256"] = _sha256_file(p)
            except Exception as e:
                rec["sha256"] = f"ERROR: {e}"
        records.append(rec)

    # Optional: build ID key table + full + slim scoring table
    if merged.get("id_build_enable"):
        src_str = merged.get("id_src") or ""
        extra_src_str = merged.get("id_extra_src") or ""
        key_out_str = merged.get("id_key_out") or ""
        full_out_str = merged.get("id_full_out") or ""
        slim_out_str = merged.get("id_slim_out") or ""
        slim_cols = merged.get("id_slim_cols") or []
        id_chunk_rows = int(
            merged.get("id_chunk_rows") or merged.get("chunk_rows") or 500_000
        )

        src_path = Path(src_str) if src_str else None
        extra_src_path = Path(extra_src_str) if extra_src_str else None
        key_out = Path(key_out_str) if key_out_str else None
        full_out = Path(full_out_str) if full_out_str else None
        slim_out = Path(slim_out_str) if slim_out_str else None

        if not src_path:
            _print("[id-build] requested but id_src is empty; skipping.")
        elif not src_path.exists():
            _print(f"[id-build] requested but source missing: {src_path}")
        elif not key_out:
            _print("[id-build] requested but id_key_out is empty; skipping.")
        else:
            stats = build_ids_from_csv(
                src=src_path,
                key_out=key_out,
                full_out=full_out,
                slim_out=slim_out,
                slim_cols=slim_cols,
                chunk_rows=id_chunk_rows,
                time_col_hint=merged["time_col"],
                extra_src=extra_src_path,
            )
            # key table record
            if key_out.exists():
                rec = dict(
                    path=str(key_out.relative_to(root))
                    if str(key_out).startswith(str(root))
                    else str(key_out),
                    bytes=key_out.stat().st_size,
                    type="csv",
                    **stats,
                )
                if merged.get("hash", False):
                    try:
                        rec["sha256"] = _sha256_file(key_out)
                    except Exception as e:
                        rec["sha256"] = f"ERROR: {e}"
                records.append(rec)
            # full table record (existing data + row_id)
            if full_out and full_out.exists():
                rec_full = dict(
                    path=str(full_out.relative_to(root))
                    if str(full_out).startswith(str(root))
                    else str(full_out),
                    bytes=full_out.stat().st_size,
                    type="csv",
                    rows=stats["rows"],
                    cols=stats["cols"] + 1,  # + row_id
                    time_min=stats["time_min"],
                    time_max=stats["time_max"],
                    lat_min=stats["lat_min"],
                    lat_max=stats["lat_max"],
                    lon_min=stats["lon_min"],
                    lon_max=stats["lon_max"],
                )
                if merged.get("hash", False):
                    try:
                        rec_full["sha256"] = _sha256_file(full_out)
                    except Exception as e:
                        rec_full["sha256"] = f"ERROR: {e}"
                records.append(rec_full)
            # slim table record
            if slim_out and slim_out.exists():
                rec_slim = dict(
                    path=str(slim_out.relative_to(root))
                    if str(slim_out).startswith(str(root))
                    else str(slim_out),
                    bytes=slim_out.stat().st_size,
                    type="csv",
                    rows=stats["rows"],
                    cols=len(slim_cols) + 1,  # row_id + requested
                    time_min=stats["time_min"],
                    time_max=stats["time_max"],
                    lat_min=stats["lat_min"],
                    lat_max=stats["lat_max"],
                    lon_min=stats["lon_min"],
                    lon_max=stats["lon_max"],
                )
                if merged.get("hash", False):
                    try:
                        rec_slim["sha256"] = _sha256_file(slim_out)
                    except Exception as e:
                        rec_slim["sha256"] = f"ERROR: {e}"
                records.append(rec_slim)

    # Write manifest
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"files": records}
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _print(f"[stage] manifest → {out_manifest}  (files={len(records)})")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
        try:
            sys.stderr.close()
        except Exception:
            pass
