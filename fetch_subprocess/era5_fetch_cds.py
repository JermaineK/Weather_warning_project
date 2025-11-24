#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
era5_fetch_cds.py
Download ERA5 NetCDFs from Copernicus Data Store (CDS), month-sliced, resumable.
Also detects/expands .zip payloads that contain .nc.

Usage (example):
  python era5_fetch_cds.py \
    --start 2025-02-01 --end 2025-05-01 \
    --area "-10,135,-25,155" \
    --hours "0..23" \
    --vars u10 v10 msl t2m \
    --product reanalysis \
    --target-dir data_era5/extracted

Requirements:
  pip install cdsapi python-dateutil tqdm
  And set your ~/.cdsapirc (or %USERPROFILE%\\.cdsapirc on Windows).
"""

import argparse, os, sys, shutil, zipfile, time, json
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil.parser import isoparse

import cdsapi
from tqdm import tqdm  # noqa: F401  (kept for future progress bars)

# ---- alias map → canonical ERA5 variable names (add more as needed) ----
VAR_ALIASES = {
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "t2m": "2m_temperature",
}

def _split_maybe(seq):
    """
    Normalize a possibly single-token list/string into a list of tokens.
    Accepts ["u10","v10"] OR ["u10 v10 msl"] OR ["u10,v10,msl"] OR "u10 v10".
    """
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
    ap = argparse.ArgumentParser(description="Download ERA5 NetCDFs from CDS (month-sliced, resumable).")
    ap.add_argument("--start", required=True, help="ISO date (YYYY-MM-DD)")
    ap.add_argument("--end",   required=True, help="ISO date (YYYY-MM-DD)")
    ap.add_argument("--area",  required=True, help='latN,lonW,latS,lonE (e.g., \"-10,135,-25,155\")')
    ap.add_argument("--hours", default="0..23", help='Hours spec: "0..23", "0,6,12,18", "0 6 12 18", or "all"')
    ap.add_argument("--vars",  nargs="+", required=True, help="ERA5 var names (aliases ok: u10 v10 msl t2m)")
    ap.add_argument("--product", default="reanalysis", choices=["reanalysis"], help="ERA5 product type")
    ap.add_argument("--dataset", default="reanalysis-era5-single-levels", help="CDS dataset name")
    ap.add_argument("--target-dir", required=True, help="Output directory root")
    ap.add_argument("--suffix", default="", help="Optional filename suffix (e.g. 'oper')")
    ap.add_argument("--overwrite", action="store_true", help="Re-download even if file exists")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests")
    ap.add_argument("--dry-run", action="store_true", help="Print built CDS request(s) and exit")
    args = ap.parse_args()

    hours = parse_hours(args.hours)
    area = parse_area(args.area)
    raw_vars = _split_maybe(args.vars)
    vars_canon = canonical_vars(raw_vars)

    out_root = Path(args.target_dir)
    ensure_dir(out_root)

    # quick sanity logs
    print("[cfg] dataset        :", args.dataset)
    print("[cfg] product_type   :", args.product)
    print("[cfg] vars (input)   :", args.vars)
    print("[cfg] vars (parsed)  :", raw_vars)
    print("[cfg] vars (canon)   :", vars_canon)
    print("[cfg] hours          :", hours)
    print("[cfg] area [N,W,S,E] :", area)

    if args.dry_run:
        # Build and show the first month's request only
        for year, month in month_span(args.start, args.end):
            yyyy = f"{year:04d}"
            mm   = f"{month:02d}"
            req = {
                "product_type": args.product,
                "variable": vars_canon,
                "year":   [yyyy],                # list!
                "month":  [mm],                  # list!
                "day":    [f"{d:02d}" for d in range(1, 32)],
                "time":   hours,
                "area":   area,
                "format": "netcdf",
            }
            print("\n[dry-run] First request JSON ↓")
            print(json.dumps({"dataset": args.dataset, "request": req}, indent=2))
            return

    c = cds_client()

    for year, month in month_span(args.start, args.end):
        yyyy = f"{year:04d}"
        mm   = f"{month:02d}"

        # Per-month folder keeps things tidy:
        out_dir = out_root / f"{yyyy}" / f"{mm}"
        ensure_dir(out_dir)

        # Build filename like: era5_YYYYMM_<suffix or vars-joined>.nc
        var_tag = "-".join(vars_canon) if len(vars_canon) <= 3 else f"{len(vars_canon)}vars"
        tag = args.suffix.strip() or var_tag
        fname = f"era5_{yyyy}{mm}_{tag}.nc"
        out_path = out_dir / fname

        if out_path.exists() and not args.overwrite:
            print(f"[skip] exists: {out_path}")
            continue

        req = {
            "product_type": args.product,
            "variable": vars_canon,
            "year":   [yyyy],                # must be list
            "month":  [mm],                  # must be list
            "day":    [f"{d:02d}" for d in range(1, 32)],   # CDS ignores overflow days
            "time":   hours,                 # already HH:00 strings
            "area":   area,                  # [N, W, S, E]
            "format": "netcdf",
        }

        tmp = out_path.with_suffix(out_path.suffix + ".part")
        try:
            print(f"[CDS] {args.dataset} {yyyy}-{mm} → {out_path}")
            # Print once per request to help debug 400s
            print(f"[CDS] request keys: {list(req.keys())}")
            cdsapi.Client().retrieve(args.dataset, req, str(tmp))
            # If we somehow got a zip, expand; else rename .part → final
            if tmp.suffix.lower() == ".zip":
                nc_outs = extract_zip_if_needed(tmp, delete_zip=True)
                if not nc_outs:
                    print(f"[warn] zip contained no .nc for {yyyy}-{mm}")
                else:
                    for p in nc_outs:
                        print(f"[ok] wrote {p}")
            else:
                tmp.replace(out_path)
                print(f"[ok] wrote {out_path}")
        except Exception as e:
            # cdsapi often embeds server message in the exception string
            print(f"[err] {yyyy}-{mm} failed.\n       {e}\n"
                  f"       Tip: re-run with --dry-run and paste the JSON into the CDS web form for "
                  f"{args.dataset}; also ensure variables are canonical.")
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

        if args.sleep > 0:
            time.sleep(args.sleep)

    print("[done] ERA5 fetch complete.")

if __name__ == "__main__":
    main()