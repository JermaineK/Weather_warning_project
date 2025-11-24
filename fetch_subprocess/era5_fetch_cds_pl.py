#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
era5_fetch_cds_pl.py
Download ERA5 *pressure-level* NetCDFs from Copernicus Data Store (CDS), month-sliced, resumable.
Supports --pl-vars and --pl-levels for pressure-level data.

Example:
  python era5_fetch_cds_pl.py \
    --start 2025-02-01 --end 2025-05-01 \
    --area "-10,135,-25,155" \
    --hours "0..23" \
    --pl-vars u v \
    --pl-levels 1000,925,850,700,500 \
    --target-dir data_era5/extracted \
    --pl-suffix uv
"""

import argparse, os, sys, shutil, zipfile, time, json
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil.parser import isoparse
import cdsapi

# ---------------- helpers ----------------

def _split_maybe(seq):
    """Accept ['u','v'] or ['u v'] or ['u,v'] or 'u v' etc."""
    if isinstance(seq, (list, tuple)):
        if len(seq) == 1 and isinstance(seq[0], str):
            s = seq[0].replace(",", " ")
            return [x for x in s.split() if x]
        return [str(x) for x in seq]
    if isinstance(seq, str):
        return [x for x in seq.replace(",", " ").split() if x]
    return [str(seq)]

def parse_hours(hspec: str) -> list[str]:
    if hspec is None:
        hrs = list(range(24))
    else:
        hspec = hspec.strip().lower()
        if hspec in ("", "all", "0..23"):
            hrs = list(range(24))
        elif ".." in hspec:
            a, b = [int(x) for x in hspec.split("..")]
            step = 1 if a <= b else -1
            hrs = list(range(a, b + step, step))
        elif "," in hspec:
            hrs = [int(x) for x in hspec.split(",") if x.strip()]
        else:
            hrs = [int(x) for x in hspec.split() if x.strip()]
    return [f"{h:02d}:00" for h in hrs]

def parse_area(aoi: str) -> list[float]:
    latN, lonW, latS, lonE = [float(x.strip()) for x in aoi.split(",")]
    return [latN, lonW, latS, lonE]

def month_span(start: str, end: str):
    t0 = isoparse(start).date().replace(day=1)
    t1 = isoparse(end).date().replace(day=1)
    cur = t0
    while cur <= t1:
        yield cur.year, cur.month
        cur = (cur + relativedelta(months=+1)).replace(day=1)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def extract_zip_if_needed(path: Path, delete_zip=True):
    outs = []
    if path.suffix.lower() == ".zip" and path.exists():
        import zipfile, shutil
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if name.lower().endswith(".nc"):
                    outp = path.with_name(Path(name).name)
                    with zf.open(name) as src, open(outp, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    outs.append(outp)
        if delete_zip:
            path.unlink(missing_ok=True)
    return outs

def cds_client():
    return cdsapi.Client()

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Download ERA5 pressure-level data from CDS.")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--area", required=True, help="latN,lonW,latS,lonE")
    ap.add_argument("--hours", default="0..23")
    ap.add_argument("--pl-vars", nargs="+", required=True, help="ERA5 pressure-level vars (aliases ok: u v)")
    ap.add_argument("--pl-levels", nargs="+", required=True, help="Pressure levels in hPa, e.g. 1000 925 850 700 500")
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--pl-suffix", default="pl")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    hours = parse_hours(args.hours)
    area = parse_area(args.area)
    pl_vars = _split_maybe(args.pl_vars)
    pl_levels = _split_maybe(args.pl_levels)
    out_root = Path(args.target_dir)
    ensure_dir(out_root)

    print("[cfg] dataset        : reanalysis-era5-pressure-levels")
    print("[cfg] vars (parsed)  :", pl_vars)
    print("[cfg] levels         :", pl_levels)
    print("[cfg] hours          :", hours)
    print("[cfg] area [N,W,S,E] :", area)

    if args.dry_run:
        y0, m0 = next(month_span(args.start, args.end))
        yyyy, mm = f"{y0:04d}", f"{m0:02d}"
        req = {
            "product_type": "reanalysis",
            "variable": pl_vars,
            "pressure_level": pl_levels,
            "year": [yyyy],
            "month": [mm],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": hours,
            "area": area,
            "format": "netcdf",
        }
        print("\n[dry-run] First request JSON ↓")
        print(json.dumps({"dataset": "reanalysis-era5-pressure-levels", "request": req}, indent=2))
        return

    c = cds_client()

    for year, month in month_span(args.start, args.end):
        yyyy = f"{year:04d}"
        mm = f"{month:02d}"
        out_dir = out_root / f"{yyyy}" / f"{mm}"
        ensure_dir(out_dir)

        fname = f"era5_pl_{yyyy}{mm}_{args.pl_suffix}.nc"
        out_path = out_dir / fname
        if out_path.exists() and not args.overwrite:
            print(f"[skip] exists: {out_path}")
            continue

        req = {
            "product_type": "reanalysis",
            "variable": pl_vars,
            "pressure_level": pl_levels,
            "year": [yyyy],
            "month": [mm],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": hours,
            "area": area,
            "format": "netcdf",
        }

        tmp = out_path.with_suffix(".part")
        try:
            print(f"[CDS] reanalysis-era5-pressure-levels {yyyy}-{mm} → {out_path}")
            c.retrieve("reanalysis-era5-pressure-levels", req, str(tmp))
            if tmp.suffix.lower() == ".zip":
                extract_zip_if_needed(tmp, delete_zip=True)
            else:
                tmp.replace(out_path)
            print(f"[ok] wrote {out_path}")
        except Exception as e:
            print(f"[err] {yyyy}-{mm} failed:\n    {e}")
            tmp.unlink(missing_ok=True)

        if args.sleep > 0:
            time.sleep(args.sleep)

    print("[done] ERA5 pressure-level fetch complete.")

if __name__ == "__main__":
    main()