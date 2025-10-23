# harvest_era5.py  — ERA5 downloader with verbose logging + sanity checks
import argparse
from pathlib import Path
import json
import time

def harvest(out_nc: Path, date_start: str, date_end: str, north: float, south: float, west: float, east: float,
            cds_url: str | None, cds_key: str | None, timeout: int = 600):
    import cdsapi

    print("== ERA5 harvest starting ==")
    print("Output:", out_nc)
    print("Date range:", date_start, "→", date_end)
    print("BBox [N,W,S,E]:", north, west, south, east)
    if cds_url and cds_key:
        print("Using explicit CDS credentials (url + key).")
        c = cdsapi.Client(url=cds_url, key=cds_key, timeout=timeout, progress=True)
    else:
        print("Using default CDS credentials (~/.cdsapirc).")
        c = cdsapi.Client(timeout=timeout, progress=True)

    req = {
        "product_type": "reanalysis",
        "variable": [
            "10m_u_component_of_wind","10m_v_component_of_wind","mean_sea_level_pressure"
        ],
        "date": f"{date_start}/{date_end}",
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [north, west, south, east],  # N, W, S, E
        "format": "netcdf",
    }
    print("Request:", json.dumps(req, indent=2))
    out_nc = Path(out_nc)
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_nc.with_suffix(out_nc.suffix + ".part")
    if tmp.exists():
        print("Removing stale partial:", tmp)
        tmp.unlink()

    t0 = time.time()
    c.retrieve("reanalysis-era5-single-levels", req, str(out_nc))
    dt = time.time() - t0

    if not out_nc.exists() or out_nc.stat().st_size == 0:
        raise RuntimeError("Download finished without a file. Check credentials, licence acceptance, or firewall.")

    print(f"== Done. Wrote {out_nc}  ({out_nc.stat().st_size/1e6:.1f} MB in {dt:.1f}s) ==")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output NetCDF path, e.g. data/era5_au_2024-12-01.nc")
    ap.add_argument("--date-start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--date-end",   required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--north", type=float, default=-10.0)
    ap.add_argument("--south", type=float, default=-25.0)
    ap.add_argument("--west",  type=float, default=135.0)
    ap.add_argument("--east",  type=float, default=155.0)
    ap.add_argument("--cds-url", default=None, help="Optional CDS URL, e.g. https://cds.climate.copernicus.eu/api")
    ap.add_argument("--cds-key", default=None, help="Optional 'uid:apikey' string")
    args = ap.parse_args()
    harvest(Path(args.out), args.date_start, args.date_end, args.north, args.south, args.west, args.east,
            args.cds_url, args.cds_key)

if __name__ == "__main__":
    main()
