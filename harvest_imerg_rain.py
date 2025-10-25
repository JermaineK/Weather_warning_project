#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
harvest_imerg_rain.py
Automated IMERG (GPM) rainfall harvest with NASA Earthdata authentication.

Modes:
  1) Grid mode  (--mode grid): AOI hourly grid (decimated with --stride)
  2) Seeds mode (--mode seeds): sample rainfall at seed points

Outputs: Parquet (rain_mmhr).

Auth:
  1) Create free Earthdata account: https://urs.earthdata.nasa.gov/
  2) Put credentials in ~/.netrc (Linux/macOS) or %USERPROFILE%\\.netrc (Windows):
       machine urs.earthdata.nasa.gov login <USERNAME> password <PASSWORD>

Products (V07):
  - Final: directory  IMERG/3B-HHR.MS.MRG.3IMERG
  - Late : directory  IMERG/3B-HHR-L.MS.MRG.3IMERG
  - Early: directory  IMERG/3B-HHR-E.MS.MRG.3IMERG
File names:
  <prefix>.<YYYYMMDD-SHHMMSS-EHHMMSS>.0000.V07B.HDF5
  where prefix is 3B-HHR[ -L | -E ].MS.MRG.3IMERG (no suffix for Final)
"""

import argparse, os, re, sys, time
from pathlib import Path
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import h5py

# ----------------------- forgiving seeds loader -----------------------

CAND_TIME = ["seed_time", "time_h", "time", "seed_start", "start_time"]
CAND_LAT  = ["seed_lat", "lat", "lat_cen", "start_lat", "y", "latitude", "LAT", "Latitude"]
CAND_LON  = ["seed_lon", "lon", "lon_cen", "start_lon", "x", "longitude", "LON", "Longitude"]
CAND_FLAG = ["flag", "alert", "alert_final", "__flag__", "__flag__auto__"]

def read_any(p, **kw):
    p = str(p)
    if p.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(p, **kw)
    return pd.read_csv(p, compression="infer", low_memory=False, **kw)

def _try_parse_time_raw(s: pd.Series, fmt: str | None = None) -> pd.Series:
    raw = s.astype(str).str.strip().str.replace("Z", "", regex=False)
    t1 = pd.to_datetime(raw, utc=True, errors="coerce")
    if t1.notna().mean() > 0.5:
        return t1.dt.tz_localize(None)
    if fmt:
        try:
            t2 = pd.to_datetime(raw, format=fmt, utc=True, errors="coerce")
            if t2.notna().mean() > 0.5:
                return t2.dt.tz_localize(None)
        except Exception:
            pass
    num = pd.to_numeric(raw, errors="coerce")
    if num.notna().any():
        mid = float(np.nanmedian(num))
        unit = "ms" if mid > 1e11 else "s"
        t3 = pd.to_datetime(num, unit=unit, utc=True, errors="coerce")
        if t3.notna().mean() > 0.5:
            return t3.dt.tz_localize(None)
    t4 = pd.to_datetime(raw, utc=True, errors="coerce", infer_datetime_format=True)
    return t4.dt.tz_localize(None)

def _norm_lon(series: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if mode in (None, "", "none"):
        return x
    if mode == "0..360":
        return (x % 360 + 360) % 360
    return ((x + 180) % 360) - 180

def _pick_col(cols_present, candidates):
    for c in candidates:
        if c in cols_present:
            return c
    lower = {c.lower(): c for c in cols_present}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def load_seeds(path, normalize_lon="none", time_format=None):
    """
    Accepts CSV/Parquet with flexible column names.
    Returns columns: time, time_h, lat, lon, lon360, flag
    """
    df = read_any(path).replace([np.inf, -np.inf], np.nan)

    tcol  = _pick_col(df.columns, CAND_TIME)
    latc  = _pick_col(df.columns, CAND_LAT)
    lonc  = _pick_col(df.columns, CAND_LON)
    flagc = _pick_col(df.columns, CAND_FLAG)

    if tcol is None:
        raise ValueError(f"{path}: need time-like column (any of {CAND_TIME}); found {list(df.columns)[:12]}...")
    if latc is None or lonc is None:
        raise ValueError(f"{path}: need lat/lon columns (lat any of {CAND_LAT}; lon any of {CAND_LON}); found {list(df.columns)[:12]}...")

    t = _try_parse_time_raw(df[tcol], time_format)
    lat = pd.to_numeric(df[latc], errors="coerce")
    lon = _norm_lon(df[lonc], normalize_lon)

    out = pd.DataFrame({"time": t, "lat": lat, "lon": lon})
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce").dt.tz_localize(None)
    out["time_h"] = out["time"].dt.floor("h")
    out["lon360"] = (out["lon"] % 360 + 360) % 360
    out["flag"] = pd.to_numeric(df[flagc], errors="coerce").fillna(0).astype(int) if flagc else 1

    out = out.dropna(subset=["time", "lat", "lon"]).drop_duplicates(["time_h", "lat", "lon"]).reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{path}: no valid rows after parsing; check column mapping and normalization.")
    return out

# ----------------------- util & product mapping -----------------------

def _ensure_dir(p: Path): p.parent.mkdir(parents=True, exist_ok=True)

def _parse_area(aoi: str | None):
    if not aoi: return None
    latN, lonW, latS, lonE = [float(x.strip()) for x in aoi.split(",")]
    return latN, lonW, latS, lonE

def hour_range(start: datetime, end: datetime):
    t = start.replace(minute=0, second=0, microsecond=0)
    while t <= end:
        yield t
        t += timedelta(hours=1)

GESDISC_HOST = "https://gpm1.gesdisc.eosdis.nasa.gov"

# Map product → (directory subpath, filename prefix)
PRODUCTS = {
    "final": ("IMERG/3B-HHR.MS.MRG.3IMERG",   "3B-HHR.MS.MRG.3IMERG"),
    "late" : ("IMERG/3B-HHR-L.MS.MRG.3IMERG", "3B-HHR-L.MS.MRG.3IMERG"),
    "early": ("IMERG/3B-HHR-E.MS.MRG.3IMERG", "3B-HHR-E.MS.MRG.3IMERG"),
}

def resolve_product(product: str):
    p = (product or "final").lower()
    if p == "auto":
        # Use Early for most recent, else Final. Simple rule: try Final first, fallback to Early/Late per file.
        # We'll still start with Final; the download helper tries alternatives if needed.
        p = "final"
    if p not in PRODUCTS:
        raise ValueError("--product must be one of: final, late, early, auto")
    return PRODUCTS[p]

def earthdata_session():
    s = requests.Session()
    s.trust_env = True
    s.headers.update({"User-Agent": "imerg-harvester/0.2"})
    return s

LIST_RE = re.compile(r'href="([^"]+\.HDF5)"')

def list_month_filenames(sess: requests.Session, dir_subpath: str, year: int, month: int) -> list[str]:
    url = f"{GESDISC_HOST}/data/{dir_subpath}/{year:04d}/{month:02d}/"
    r = sess.get(url)
    if r.status_code != 200:
        raise RuntimeError(f"List HTTP {r.status_code} for {url}. Earthdata login configured?")
    files = LIST_RE.findall(r.text)
    return files  # just names

def build_halfhour_name(prefix: str, t: datetime, vtag="V07B"):
    """
    Return two candidate names for the half-hour starting at t:
      …-EHHMMSS with :29:59 (canonical) and :30:00 (fallback).
    """
    s_stamp = t.strftime("%Y%m%d-S%H%M%S")
    e1 = (t + timedelta(minutes=30) - timedelta(seconds=1)).strftime("E%H%M%S")  # ...2959
    e2 = (t + timedelta(minutes=30)).strftime("E%H%M%S")                          # ...3000
    name1 = f"{prefix}.{s_stamp}-{e1}.0000.{vtag}.HDF5"
    name2 = f"{prefix}.{s_stamp}-{e2}.0000.{vtag}.HDF5"
    return [name1, name2]

def pick_granule_for_start(sess: requests.Session, dir_subpath: str, prefix: str, t: datetime):
    """
    List month and pick the first matching name that exists (2959 or 3000).
    Returns (used_name).
    """
    names = list_month_filenames(sess, dir_subpath, t.year, t.month)
    cands = build_halfhour_name(prefix, t)
    for nm in cands:
        if nm in names:
            return nm
    # As a fallback, allow Late/Early alternatives if we started with Final
    if prefix == "3B-HHR.MS.MRG.3IMERG":
        for alt_dir, alt_pref in [PRODUCTS["late"], PRODUCTS["early"]]:
            try:
                alt_names = list_month_filenames(sess, alt_dir, t.year, t.month)
                for nm in build_halfhour_name(alt_pref, t):
                    if nm in alt_names:
                        return nm
            except Exception:
                pass
    raise FileNotFoundError(f"No IMERG file found for half-hour starting {t.isoformat()} in {dir_subpath} (tried {cands})")

def download_if_missing(sess, url: str, out_path: Path, retries=3):
    if out_path.exists() and out_path.stat().st_size > 0:
        return False
    _ensure_dir(out_path)
    for k in range(retries):
        try:
            with sess.get(url, stream=True) as r:
                r.raise_for_status()
                tmp = out_path.with_suffix(out_path.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1<<20):
                        if chunk:
                            f.write(chunk)
                tmp.replace(out_path)
                return True
        except Exception:
            if k == retries - 1: raise
            time.sleep(2 + 2*k)
    return False

# ----------------------- reading IMERG -----------------------

def open_imerg_subset(path: Path, area=None, stride=1):
    """
    Return (lat, lon, rain_mmhr) arrays subset to AOI (lon in 0..360 grid).
    """
    with h5py.File(path, "r") as h5:
        rain = h5["/Grid/precipitationCal"][:]  # mm/hr
        lat = h5["/Grid/lat"][:]
        lon = h5["/Grid/lon"][:]
        if area:
            latN, lonW, latS, lonE = area
            wrap360 = lambda x: (x % 360 + 360) % 360
            lonW360, lonE360 = wrap360(lonW), wrap360(lonE)
            Lmask = (lat <= latN) & (lat >= latS)
            if lonW360 <= lonE360:
                Smask = (lon >= lonW360) & (lon <= lonE360)
            else:
                Smask = (lon >= lonW360) | (lon <= lonE360)
            lat_idx = np.where(Lmask)[0][::stride]
            lon_idx = np.where(Smask)[0][::stride]
        else:
            lat_idx = np.arange(len(lat))[::stride]
            lon_idx = np.arange(len(lon))[::stride]
        sub = rain[np.ix_(lat_idx, lon_idx)]
        return lat[lat_idx], lon[lon_idx], sub

def nearest_index(arr, value): return int(np.abs(arr - value).argmin())

# ----------------------- core harvest -----------------------

def grid_mode(cache_dir: Path, out_parquet: Path, start: datetime, end: datetime, area, stride: int,
              product: str = "final"):
    dir_sub, prefix = resolve_product(product)
    sess = earthdata_session()
    rows = []
    for hr in hour_range(start, end):
        # first and second half-hour names
        n1 = pick_granule_for_start(sess, dir_sub, prefix, hr)
        n2 = pick_granule_for_start(sess, dir_sub, prefix, hr + timedelta(minutes=30))
        u1 = f"{GESDISC_HOST}/data/{dir_sub}/{hr.year:04d}/{hr.month:02d}/{n1}"
        t2 = hr + timedelta(minutes=30)
        u2 = f"{GESDISC_HOST}/data/{dir_sub}/{t2.year:04d}/{t2.month:02d}/{n2}"

        p1 = cache_dir / dir_sub / f"{hr.year:04d}" / f"{hr.month:02d}" / n1
        p2 = cache_dir / dir_sub / f"{t2.year:04d}" / f"{t2.month:02d}" / n2
        download_if_missing(sess, u1, p1)
        download_if_missing(sess, u2, p2)

        latA, lonA, rA = open_imerg_subset(p1, area, stride)
        latB, lonB, rB = open_imerg_subset(p2, area, stride)
        if not (np.array_equal(latA, latB) and np.array_equal(lonA, lonB)):
            raise RuntimeError("IMEGR lat/lon mismatch across consecutive files.")
        rH = np.nanmean(np.stack([rA, rB], axis=0), axis=0)
        LAT, LON = np.meshgrid(latA, lonA, indexing="ij")
        df = pd.DataFrame({
            "time": hr.replace(tzinfo=None),
            "lat": LAT.ravel(),
            "lon": ((LON.ravel() + 180) % 360) - 180,
            "rain_mmhr": rH.ravel(),
        })
        rows.append(df)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["time","lat","lon","rain_mmhr"])
    _ensure_dir(out_parquet)
    out.to_parquet(out_parquet, index=False)
    print(f"[grid] wrote {out_parquet} | rows={len(out):,}")

def seeds_mode(cache_dir: Path, seeds_path: Path, out_parquet: Path, start: datetime, end: datetime,
               radius_km: float, agg: str, normalize_lon="none", product: str = "final"):
    seeds = load_seeds(seeds_path, normalize_lon=normalize_lon)
    # Restrict to window
    if start: seeds = seeds[seeds["time"] >= start.replace(tzinfo=None)]
    if end:   seeds = seeds[seeds["time"] <= end.replace(tzinfo=None)]
    if seeds.empty:
        print("[seeds] no rows in time window.")
        _ensure_dir(out_parquet)
        pd.DataFrame(columns=["time","lat","lon","rain_mmhr"]).to_parquet(out_parquet, index=False)
        return

    dir_sub, prefix = resolve_product(product)
    sess = earthdata_session()
    rows = []
    grid_lat, grid_lon = None, None

    for hr, g in seeds.groupby("time_h"):
        # pick file names from directory listing; this avoids 404s
        n1 = pick_granule_for_start(sess, dir_sub, prefix, hr)
        n2 = pick_granule_for_start(sess, dir_sub, prefix, hr + timedelta(minutes=30))
        u1 = f"{GESDISC_HOST}/data/{dir_sub}/{hr.year:04d}/{hr.month:02d}/{n1}"
        t2 = hr + timedelta(minutes=30)
        u2 = f"{GESDISC_HOST}/data/{dir_sub}/{t2.year:04d}/{t2.month:02d}/{n2}"

        p1 = cache_dir / dir_sub / f"{hr.year:04d}" / f"{hr.month:02d}" / n1
        p2 = cache_dir / dir_sub / f"{t2.year:04d}" / f"{t2.month:02d}" / n2
        download_if_missing(sess, u1, p1)
        download_if_missing(sess, u2, p2)

        with h5py.File(p1, "r") as h5a, h5py.File(p2, "r") as h5b:
            ra = h5a["/Grid/precipitationCal"][:]
            rb = h5b["/Grid/precipitationCal"][:]
            rH = np.nanmean(np.stack([ra, rb], axis=0), axis=0)
            if grid_lat is None:
                grid_lat = h5a["/Grid/lat"][:]
                grid_lon = h5a["/Grid/lon"][:]

        for _, s in g.iterrows():
            i = nearest_index(grid_lat, s["lat"])
            j = nearest_index(grid_lon, s["lon360"])
            if agg == "nearest":
                val = float(rH[i, j])
            else:  # mean3x3 window
                i0, i1 = max(0, i-1), min(len(grid_lat)-1, i+1)
                j0, j1 = max(0, j-1), min(len(grid_lon)-1, j+1)
                val = float(np.nanmean(rH[i0:i1+1, j0:j1+1]))
            rows.append({
                "time": hr.replace(tzinfo=None),
                "lat": float(s["lat"]),
                "lon": float(s["lon"]),
                "rain_mmhr": val,
            })

    out = pd.DataFrame(rows)
    _ensure_dir(out_parquet)
    out.to_parquet(out_parquet, index=False)
    print(f"[seeds] wrote {out_parquet} | rows={len(out):,}")

# ----------------------- CLI -----------------------

def main():
    ap = argparse.ArgumentParser(description="Automated IMERG rainfall harvest (NASA GES DISC).")
    ap.add_argument("--mode", choices=["grid","seeds"], default="seeds")
    ap.add_argument("--cache-dir", default="data_imerg/cache")
    ap.add_argument("--start", required=True, help="UTC start (e.g., 2025-02-01)")
    ap.add_argument("--end",   required=True, help="UTC end   (e.g., 2025-05-01)")
    ap.add_argument("--product", choices=["final","late","early","auto"], default="final",
                    help="IMERG run to use (auto tries Final, falls back per file).")
    # grid
    ap.add_argument("--area", default=None, help="AOI latN,lonW,latS,lonE (grid mode)")
    ap.add_argument("--stride", type=int, default=2, help="Grid decimation (1=no decimation)")
    # seeds
    ap.add_argument("--seeds", default=None, help="Seeds CSV/Parquet (seeds mode)")
    ap.add_argument("--radius-km", type=float, default=15.0, help="Sampling radius (~3x3)")
    ap.add_argument("--agg", choices=["nearest","mean3x3"], default="mean3x3")
    ap.add_argument("--normalize-lon", choices=["none","-180..180","0..360"], default="none")
    # out
    ap.add_argument("--out", required=True, help="Output Parquet path")
    args = ap.parse_args()

    start = pd.to_datetime(args.start, utc=True).to_pydatetime().replace(tzinfo=timezone.utc)
    end   = pd.to_datetime(args.end,   utc=True).to_pydatetime().replace(tzinfo=timezone.utc)
    area = _parse_area(args.area) if args.area else None

    cache_dir = Path(args.cache_dir)
    out_path  = Path(args.out)

    if args.mode == "grid":
        if not area:
            raise ValueError("--area required in grid mode.")
        grid_mode(cache_dir, out_path, start, end, area, args.stride, product=args.product)
    else:
        if not args.seeds:
            raise ValueError("--seeds required in seeds mode.")
        seeds_mode(cache_dir, Path(args.seeds), out_path, start, end,
                   args.radius_km, args.agg, normalize_lon=args.normalize_lon, product=args.product)

if __name__ == "__main__":
    main()