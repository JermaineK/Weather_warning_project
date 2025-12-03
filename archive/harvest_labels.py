#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
harvest_labels.py
Create storm label CSVs for Australia (or any AOI) from:
  (A) IBTrACS best tracks (historical, via CSV or local file)
  (B) optional BoM live warnings (current only)

Outputs a tidy CSV: time, lat, lon, label, source, details

Improvements:
- Chunked & schema-tolerant IBTrACS read (fast, low memory)
- Optional local path or URL with on-disk cache and --force-refresh
- Robust time parsing and hour-floor in UTC
- Longitude normalization and anti-meridian-aware AOI crop
- Configurable BoM grid step and HTTP timeouts
"""

from __future__ import annotations
import argparse, csv, json, math, os, sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Iterable, Tuple, Optional
import urllib.request as ureq
import urllib.error as uerr
import pandas as pd
import numpy as np

DEFAULT_IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
)
BOM_WARNINGS_URL = "https://api.weather.bom.gov.au/v1/warnings"  # live only

# Default Australia bbox (south, west, north, east) in lon ∈ [-180,180]
DEFAULT_AU_BBOX = (-45.0, 110.0, -10.0, 160.0)

# ----------------------------- utils -----------------------------

def to_utc_hour_floor(s: pd.Series) -> pd.Series:
    """Parse to UTC naive and floor to the hour."""
    t = pd.to_datetime(s, utc=True, errors="coerce")
    return t.dt.tz_localize(None).dt.floor("h")

def norm_lon(x: pd.Series, mode: str) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce")
    if mode == "none":
        return v
    if mode == "0..360":
        return (v % 360 + 360) % 360
    # default: -180..180
    return ((v + 180) % 360) - 180

def crosses_antimeridian(w: float, e: float) -> bool:
    return w > e

def in_bbox(lat: float, lon: float, bbox: Tuple[float, float, float, float]) -> bool:
    s, w, n, e = bbox
    if not (s <= lat <= n):
        return False
    if crosses_antimeridian(w, e):
        return lon >= w or lon <= e
    return w <= lon <= e

def month_range(year: int, month: int):
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end = (datetime(year + (month==12), (month % 12) + 1, 1, tzinfo=timezone.utc))
    return start, end

def frange(a: float, b: float, step: float) -> Iterable[float]:
    x = a
    # safeguard against float drift
    while x <= b + 1e-9:
        yield float(round(x, 6))
        x += step

def http_download(url: str, dest: Path, timeout: float = 20.0, force: bool = False):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    print(f"[download] {url} -> {dest}")
    req = ureq.Request(url, headers={"User-Agent": "pixel-theory-harvest/1.0"})
    with ureq.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        f.write(r.read())
    # quick sanity check
    if dest.stat().st_size < 1024:  # <1 KB is suspicious for IBTrACS
        raise RuntimeError(f"Downloaded file too small: {dest}")
    return dest

def parse_area(aoi: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not aoi:
        return None
    parts = [p.strip() for p in aoi.split(",")]
    if len(parts) != 4:
        raise ValueError("--area must be 'south,west,north,east'")
    return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])

# ----------------------- IBTrACS harvest ------------------------

IB_COLS_PREF = {
    "time": ["ISO_TIME", "iso_time", "time", "datetime"],
    "lat":  ["LAT", "lat", "Latitude"],
    "lon":  ["LON", "lon", "Longitude"],
    # wind/pressure: tolerant to agency variants; optional
    "wind": ["USA_WIND", "WMO_WIND", "usa_wind", "wmo_wind", "WIND", "vmax"],
    "pres": ["USA_PRES", "WMO_PRES", "usa_pres", "wmo_pres", "PRES", "pmin"],
    "sid":  ["SID", "sid", "id"],
    "name": ["NAME", "name", "storm_name", "NAME_WMO", "NAME_USA"],
}

def pick_first(df: pd.DataFrame, cands) -> Optional[str]:
    lc = {c.lower(): c for c in df.columns}
    for nm in cands:
        if nm in df.columns:
            return nm
        if nm.lower() in lc:
            return lc[nm.lower()]
    return None

def harvest_ibtracs(
    ibtracs_source: str,
    cache_path: Path,
    year: int,
    month: int,
    bbox: Tuple[float, float, float, float],
    normalize_lon_mode: str = "-180..180",
    force_refresh: bool = False,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """
    Load IBTrACS from URL or local file. Filters to given month & AOI.
    Returns rows with time/lat/lon (inside AOI) as labels=1.
    """
    # Resolve source
    src = Path(ibtracs_source)
    if src.exists():
        ib_path = src
        print(f"[ibtracs] using local file: {ib_path}")
    else:
        ib_path = http_download(ibtracs_source, cache_path, force=force_refresh)

    # Time window
    start, end = month_range(year, month)

    # Iterate in chunks (works for CSV only). If Parquet is provided locally, read once.
    if ib_path.suffix.lower() in {".parquet", ".parq", ".pq"}:
        df = pd.read_parquet(ib_path)
        return _filter_ibtracs_frame(df, start, end, bbox, normalize_lon_mode)

    # CSV streaming
    labels = []
    used_cols = None
    for chunk in pd.read_csv(ib_path, chunksize=chunksize, low_memory=False):
        if used_cols is None:
            used_cols = {k: pick_first(chunk, v) for k, v in IB_COLS_PREF.items()}
            # minimally require time/lat/lon
            if not all(used_cols.get(k) for k in ["time", "lat", "lon"]):
                raise ValueError("IBTrACS: cannot find time/lat/lon columns.")
            print("[ibtracs] columns:",
                  {k: used_cols[k] for k in ["time","lat","lon","wind","pres","sid","name"]})

        t = to_utc_hour_floor(chunk[used_cols["time"]])
        mask_time = (t >= pd.Timestamp(start.replace(tzinfo=None))) & (t < pd.Timestamp(end.replace(tzinfo=None)))

        la = pd.to_numeric(chunk[used_cols["lat"]], errors="coerce")
        lo = norm_lon(chunk[used_cols["lon"]], normalize_lon_mode)

        # AOI prefilter (pad to catch near-coast)
        s, w, n, e = bbox
        pad = 2.0
        if crosses_antimeridian(w, e):
            mask_box = (la.between(s - pad, n + pad)) & ((lo >= w - pad) | (lo <= e + pad))
        else:
            mask_box = (la.between(s - pad, n + pad)) & (lo.between(w - pad, e + pad))

        keep = chunk.loc[mask_time & mask_box].copy()
        if keep.empty:
            continue

        # Now enforce final AOI (no pad) and build rows
        t_keep = to_utc_hour_floor(keep[used_cols["time"]])
        la_keep = pd.to_numeric(keep[used_cols["lat"]], errors="coerce")
        lo_keep = norm_lon(keep[used_cols["lon"]], normalize_lon_mode)

        name_col = used_cols.get("name")
        sid_col  = used_cols.get("sid")

        for tt, la0, lo0, nm, sid in zip(t_keep, la_keep, lo_keep,
                                         (keep[name_col] if name_col in keep else [None]*len(keep)),
                                         (keep[sid_col] if sid_col in keep else [None]*len(keep))):
            if pd.isna(tt) or pd.isna(la0) or pd.isna(lo0):
                continue
            if in_bbox(float(la0), float(lo0), bbox):
                details = f"{sid}|{nm}" if (sid is not None or nm is not None) else ""
                labels.append({
                    "time": tt.strftime("%Y-%m-%dT%H:00:00Z"),
                    "lat": float(la0), "lon": float(lo0),
                    "label": 1,
                    "source": "IBTrACS",
                    "details": details
                })

    df_out = pd.DataFrame(labels)
    if not df_out.empty:
        # de-dup just in case
        df_out = df_out.drop_duplicates(subset=["time","lat","lon","source"]).reset_index(drop=True)
    return df_out

def _filter_ibtracs_frame(df: pd.DataFrame, start, end, bbox, normalize_lon_mode):
    used = {k: pick_first(df, v) for k, v in IB_COLS_PREF.items()}
    t = to_utc_hour_floor(df[used["time"]])
    la = pd.to_numeric(df[used["lat"]], errors="coerce")
    lo = norm_lon(df[used["lon"]], normalize_lon_mode)
    mask_time = (t >= pd.Timestamp(start.replace(tzinfo=None))) & (t < pd.Timestamp(end.replace(tzinfo=None)))

    rows = []
    for tt, la0, lo0, sid, nm in zip(
        t[mask_time], la[mask_time], lo[mask_time],
        (df[used["sid"]][mask_time] if used.get("sid") in df else [None]*int(mask_time.sum())),
        (df[used["name"]][mask_time] if used.get("name") in df else [None]*int(mask_time.sum())),
    ):
        if pd.isna(tt) or pd.isna(la0) or pd.isna(lo0):
            continue
        if in_bbox(float(la0), float(lo0), bbox):
            rows.append({
                "time": tt.strftime("%Y-%m-%dT%H:00:00Z"),
                "lat": float(la0), "lon": float(lo0),
                "label": 1, "source": "IBTrACS",
                "details": f"{sid}|{nm}" if (sid is not None or nm is not None) else ""
            })
    return pd.DataFrame(rows)

# ----------------------- BoM live (optional) -----------------------

def harvest_bom_live(
    bbox: Tuple[float,float,float,float],
    grid_step_deg: float = 2.0,
    timeout: float = 10.0,
) -> pd.DataFrame:
    """BoM warnings lack precise geometry; tag a coarse grid over AU when warnings exist."""
    try:
        req = ureq.Request(BOM_WARNINGS_URL, headers={"User-Agent": "pixel-theory-harvest/1.0"})
        with ureq.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (uerr.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"[bom] fetch failed: {e}")
        return pd.DataFrame(columns=["time","lat","lon","label","source","details"])

    items = data.get("warnings", []) if isinstance(data, dict) else []
    has_severe = any(
        "severe" in (w.get("headline","") or "").lower()
        or "thunderstorm" in (w.get("headline","") or "").lower()
        for w in items
    )
    if not has_severe:
        return pd.DataFrame(columns=["time","lat","lon","label","source","details"])

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    s, w, n, e = bbox
    lats = list(frange(s, n, grid_step_deg))
    lons = list(frange(w, e, grid_step_deg)) if not crosses_antimeridian(w, e) else \
           list(frange(w, 180.0, grid_step_deg)) + list(frange(-180.0, e, grid_step_deg))
    rows = []
    for la in lats:
        for lo in lons:
            if in_bbox(la, lo, bbox):
                rows.append({
                    "time": now.strftime("%Y-%m-%dT%H:00:00Z"),
                    "lat": la, "lon": lo,
                    "label": 1,
                    "source": "BOM_live",
                    "details": "active_severe_warning"
                })
    return pd.DataFrame(rows)

# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Harvest storm labels (IBTrACS + optional BoM live).")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--out", default=None)

    # Sources & caching
    ap.add_argument("--ibtracs", default=DEFAULT_IBTRACS_URL,
                    help="IBTrACS CSV URL or local file path.")
    ap.add_argument("--cache-path", default="data/_cache/ibtracs.csv",
                    help="Cache file for remote IBTrACS CSV.")
    ap.add_argument("--force-refresh", action="store_true", help="Re-download remote file.")

    # Geometry
    ap.add_argument("--area", default=None, help="south,west,north,east (lon in chosen normalization)")
    ap.add_argument("--normalize-lon", choices=["-180..180","0..360","none"], default="-180..180")

    # BoM live
    ap.add_argument("--include-bom-live", action="store_true")
    ap.add_argument("--bom-grid-step", type=float, default=2.0)

    args = ap.parse_args()

    Path("data").mkdir(parents=True, exist_ok=True)
    out = args.out or f"data/storm_labels_{args.year:04d}-{args.month:02d}.csv"

    bbox = parse_area(args.area) or DEFAULT_AU_BBOX

    # IBTrACS
    ib = harvest_ibtracs(
        ibtracs_source=args.ibtracs,
        cache_path=Path(args.cache_path),
        year=args.year,
        month=args.month,
        bbox=bbox,
        normalize_lon_mode=args.normalize_lon,
        force_refresh=args.force_refresh,
    )

    # BoM (optional)
    if args.include_bom_live:
        bl = harvest_bom_live(bbox=bbox, grid_step_deg=args.bom_grid_step)
        lab = pd.concat([ib, bl], ignore_index=True)
    else:
        lab = ib

    if lab.empty:
        print("[harvest] No labels found for the given month/filters.")
    else:
        # De-duplicate just in case
        lab = lab.drop_duplicates(subset=["time","lat","lon","source"]).reset_index(drop=True)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    lab.to_csv(out, index=False)
    print(f"[harvest] wrote {out}  rows={len(lab):,}")

if __name__ == "__main__":
    main()