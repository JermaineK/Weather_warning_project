#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
era5_fetch_cds.py
Download ERA5 NetCDFs from Copernicus Data Store (CDS), month-sliced, resumable.

Supports:
  • Single-levels:   --vars ...           (dataset: reanalysis-era5-single-levels)
  • Pressure-levels: --pl-vars --pl-levels (dataset: reanalysis-era5-pressure-levels)
You may request either or both in a single run (used by fetch_manager presets: era5, era5-pl, era5-both).

Usage (example, single-levels only):
  python era5_fetch_cds.py \
    --start 2025-02-01 --end 2025-05-01 \
    --area "-10,135,-25,155" \
    --hours "0..23" \
    --vars u10 v10 msl t2m \
    --product reanalysis \
    --target-dir data_era5/extracted

Pressure-levels only (e.g. used under 'era5-pl'):
  python era5_fetch_cds.py \
    --start 2025-02-01 --end 2025-05-01 \
    --area "-10,135,-25,155" \
    --hours "0..23" \
    --pl-vars u v \
    --pl-levels "1000,925,850,700,500" \
    --target-dir data_era5/extracted

Requirements:
  pip install cdsapi python-dateutil tqdm
  And set your ~/.cdsapirc (or %USERPROFILE%\\.cdsapirc on Windows).
"""

import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import cdsapi
from dateutil.parser import isoparse
from dateutil.relativedelta import relativedelta
from tqdm import tqdm  # noqa: F401  (kept for future progress bars)

# ---- alias map -> canonical ERA5 variable names (add more as needed) ----
VAR_ALIASES = {
    # single-levels
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "tcwv": "total_column_water_vapour",
    "tp": "total_precipitation",
    "sshf": "surface_sensible_heat_flux",
    "slhf": "surface_latent_heat_flux",
    # convective energy (tests the GSE "Energy" leg with real CAPE rather than
    # the msl/t2m-derived E_energy proxy, which showed no discrimination)
    "cape": "convective_available_potential_energy",
    "cin":  "convective_inhibition",
    # pressure-level aliases (handy)
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "z": "geopotential",
}


def _split_maybe(seq):
    """
    Normalize a possibly single-token list/string into a list of tokens.
    Accepts ["u10","v10"] OR ["u10 v10 msl"] OR ["u10,v10,msl"] OR "u10 v10".
    """
    if seq is None:
        return []
    if isinstance(seq, (list, tuple)):
        if len(seq) == 1 and isinstance(seq[0], str):
            s = seq[0].replace(",", " ")
            return [x for x in s.split() if x]
        return [str(x) for x in seq]
    if isinstance(seq, str):
        return [x for x in seq.replace(",", " ").split() if x]
    return [str(seq)]


def canonical_vars(vnames):
    out = []
    for v in vnames or []:
        key = str(v).strip()
        out.append(VAR_ALIASES.get(key.lower(), key))
    return out


# ---------- helpers ----------

def parse_hours(hspec: str) -> list[str]:
    """
    Accept "0..23", "0,6,12,18", "0 6 12 18", or "all".
    Returns zero-padded hour strings as CDS expects, e.g. "06:00".
    """
    if hspec is None:
        hrs = list(range(24))
    else:
        hspec = hspec.strip().lower()
        if hspec in ("", "all", "0..23"):
            hrs = list(range(24))
        elif ".." in hspec:
            a, b = hspec.split("..")
            a, b = int(a), int(b)
            step = 1 if a <= b else -1
            hrs = list(range(a, b + step, step))
        elif "," in hspec:
            hrs = [int(x) for x in hspec.split(",") if x.strip() != ""]
        else:
            hrs = [int(x) for x in hspec.split() if x.strip() != ""]
    return [f"{h:02d}:00" for h in hrs]


def parse_area(aoi: str) -> list[float]:
    """
    CDS wants [north, west, south, east] in -180..180. You pass "latN,lonW,latS,lonE".
    """
    latN, lonW, latS, lonE = [float(x.strip()) for x in aoi.split(",")]
    return [latN, lonW, latS, lonE]


def month_span(start: str, end: str):
    """Yield (YYYY, MM) tuples from start..end inclusive."""
    t0 = isoparse(start).date().replace(day=1)
    t1 = isoparse(end).date().replace(day=1)
    cur = t0
    while cur <= t1:
        yield cur.year, cur.month
        cur = (cur + relativedelta(months=+1)).replace(day=1)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def extract_zip_if_needed(path: Path, delete_zip: bool = True) -> list[Path]:
    """If `path` is a .zip, extract .nc files next to it. Return list of extracted .nc paths."""
    outs = []
    if path.suffix.lower() == ".zip" and path.exists():
        try:
            with zipfile.ZipFile(path, "r") as zf:
                for name in zf.namelist():
                    if name.lower().endswith(".nc"):
                        outp = path.with_name(Path(name).name)
                        with zf.open(name) as src, open(outp, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        outs.append(outp)
            if delete_zip:
                path.unlink(missing_ok=True)
        except zipfile.BadZipFile:
            pass
    return outs


def cds_client():
    # Let cdsapi read ~/.cdsapirc or %USERPROFILE%\.cdsapirc (Windows)
    return cdsapi.Client()


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Download ERA5 NetCDFs from CDS (month-sliced, resumable; single- and pressure-levels)."
    )
    ap.add_argument("--start", required=True, help="ISO date (YYYY-MM-DD)")
    ap.add_argument("--end",   required=True, help="ISO date (YYYY-MM-DD)")
    ap.add_argument("--area",  required=True, help='latN,lonW,latS,lonE (e.g., "-10,135,-25,155")')
    ap.add_argument("--hours", default="0..23",
                    help='Hours spec: "0..23", "0,6,12,18", "0 6 12 18", or "all"')

    # Single-level block (used by fetch_manager 'era5' and 'era5-both')
    ap.add_argument("--vars",  nargs="+", help="Single-level ERA5 var names (aliases ok: u10 v10 msl t2m ...)")
    ap.add_argument("--product", default="reanalysis",
                    choices=["reanalysis"], help="ERA5 product type for single-levels")
    ap.add_argument("--dataset", default="reanalysis-era5-single-levels",
                    help="CDS dataset name for single-levels")
    ap.add_argument("--suffix", default="", help="Optional filename suffix for single-level files")

    # Pressure-level block (used by fetch_manager 'era5-pl' and 'era5-both')
    ap.add_argument("--pl-vars", nargs="+",
                    help="Pressure-level ERA5 var names (aliases ok: u v z, etc.)")
    ap.add_argument("--pl-levels", default=None,
                    help='Comma/space-separated pressure levels in hPa, e.g. "1000,925,850,700,500"')
    ap.add_argument("--pl-product", default="reanalysis",
                    choices=["reanalysis"], help="ERA5 product type for pressure-levels")
    ap.add_argument("--pl-dataset", default="reanalysis-era5-pressure-levels",
                    help="CDS dataset name for pressure-levels")
    ap.add_argument("--pl-suffix", default="pl",
                    help="Filename suffix for pressure-level files (e.g. 'uv')")

    ap.add_argument("--target-dir", required=True, help="Output directory root")
    ap.add_argument("--overwrite", action="store_true", help="Re-download even if file exists")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests")
    ap.add_argument("--dry-run", action="store_true", help="Print built CDS request(s) and exit")

    args = ap.parse_args()

    hours = parse_hours(args.hours)
    area = parse_area(args.area)

    # Parse and canon single-level vars
    raw_single = _split_maybe(args.vars)
    vars_single = canonical_vars(raw_single)

    # Parse and canon pressure-level vars
    raw_pl = _split_maybe(args.pl_vars)
    vars_pl = canonical_vars(raw_pl)

    if not vars_single and not vars_pl:
        ap.error("You must provide at least one of --vars (single-levels) or --pl-vars (pressure-levels).")

    if vars_pl and not args.pl_levels:
        ap.error("You provided --pl-vars but no --pl-levels. Example: --pl-levels '1000,925,850,700,500'")

    if args.pl_levels:
        pl_levels = [s for s in args.pl_levels.replace(",", " ").split() if s]
    else:
        pl_levels = []

    out_root = Path(args.target_dir)
    ensure_dir(out_root)

    # quick sanity logs
    print("[cfg] hours          :", hours)
    print("[cfg] area [N,W,S,E] :", area)
    if vars_single:
        print("[cfg] single-level dataset :", args.dataset)
        print("[cfg] single product_type   :", args.product)
        print("[cfg] vars_single (input)   :", args.vars)
        print("[cfg] vars_single (parsed)  :", raw_single)
        print("[cfg] vars_single (canon)   :", vars_single)
    if vars_pl:
        print("[cfg] pl dataset            :", args.pl_dataset)
        print("[cfg] pl product_type       :", args.pl_product)
        print("[cfg] vars_pl (input)       :", args.pl_vars)
        print("[cfg] vars_pl (parsed)      :", raw_pl)
        print("[cfg] vars_pl (canon)       :", vars_pl)
        print("[cfg] pl_levels             :", pl_levels)

    # Build "jobs" list: one for single-levels, one for pl (optional)
    jobs = []
    if vars_single:
        jobs.append({
            "kind": "single",
            "dataset": args.dataset,
            "product_type": args.product,
            "vars": vars_single,
            "suffix": args.suffix.strip(),
        })
    if vars_pl:
        jobs.append({
            "kind": "pl",
            "dataset": args.pl_dataset,
            "product_type": args.pl_product,
            "vars": vars_pl,
            "levels": pl_levels,
            "suffix": args.pl_suffix.strip() or "pl",
        })

    if args.dry_run:
        # Show one request per job for the first month only
        for year, month in month_span(args.start, args.end):
            yyyy = f"{year:04d}"
            mm = f"{month:02d}"
            days = [f"{d:02d}" for d in range(1, 32)]
            for job in jobs:
                req = {
                    "product_type": job["product_type"],
                    "year": [yyyy],
                    "month": [mm],
                    "day": days,
                    "time": hours,
                    "area": area,
                    "format": "netcdf",
                }
                req["variable"] = job["vars"]
                if job["kind"] == "pl":
                    req["pressure_level"] = job["levels"]
                print("\n[dry-run]", job["kind"], "request JSON")
                print(json.dumps({"dataset": job["dataset"], "request": req}, indent=2))
            # Only first month for dry-run
            return

    c = cds_client()

    for year, month in month_span(args.start, args.end):
        yyyy = f"{year:04d}"
        mm = f"{month:02d}"
        days = [f"{d:02d}" for d in range(1, 32)]

        out_dir = out_root / yyyy / mm
        ensure_dir(out_dir)

        for job in jobs:
            if job["kind"] == "single":
                # filename like: era5_YYYYMM_<suffix or vars-joined>.nc
                if job["suffix"]:
                    tag = job["suffix"]
                else:
                    var_tag = "-".join(job["vars"]) if len(job["vars"]) <= 3 else f"{len(job['vars'])}vars"
                    tag = var_tag
                prefix = "era5"
            else:
                # pressure-level tag is more fixed/short (e.g. 'uv') and keeps a pl_ prefix
                tag = job["suffix"] or "pl"
                prefix = "era5_pl"

            fname = f"{prefix}_{yyyy}{mm}_{tag}.nc"
            out_path = out_dir / fname

            if out_path.exists() and not args.overwrite:
                print(f"[skip] exists: {out_path}")
                continue

            req = {
                "product_type": job["product_type"],
                "variable": job["vars"],
                "year": [yyyy],
                "month": [mm],
                "day": days,
                "time": hours,
                "area": area,
                "format": "netcdf",
            }
            if job["kind"] == "pl":
                req["pressure_level"] = job["levels"]

            tmp = out_path.with_suffix(out_path.suffix + ".part")
            try:
                print(f"[CDS] {job['kind']} {job['dataset']} {yyyy}-{mm} -> {out_path}")
                print(f"[CDS] request keys: {list(req.keys())}")
                c.retrieve(job["dataset"], req, str(tmp))

                # If we somehow got a zip, expand; else rename .part -> final
                if tmp.suffix.lower() == ".zip":
                    nc_outs = extract_zip_if_needed(tmp, delete_zip=True)
                    if not nc_outs:
                        print(f"[warn] zip contained no .nc for {yyyy}-{mm} ({job['kind']})")
                    else:
                        for p in nc_outs:
                            print(f"[ok] wrote {p}")
                else:
                    tmp.replace(out_path)
                    print(f"[ok] wrote {out_path}")
            except Exception as e:
                print(
                    f"[err] {job['kind']} {yyyy}-{mm} failed.\n"
                    f"       {e}\n"
                    f"       Tip: re-run with --dry-run and paste the JSON into the CDS web form for "
                    f"{job['dataset']}; also ensure variables and pressure levels are canonical."
                )
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass

            if args.sleep > 0:
                time.sleep(args.sleep)

    print("[done] ERA5 fetch complete.")


if __name__ == "__main__":
    main()