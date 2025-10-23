# harvest_labels.py
# Creates storm label CSVs for Australia from (A) IBTrACS best tracks (historical) and
# (B) optional BoM live warnings (current only).
# Output: data/storm_labels_YYYY-MM.csv with columns: time, lat, lon, label, source, details

from __future__ import annotations
import argparse, csv, json, math, os
from pathlib import Path
from datetime import datetime, timedelta, timezone
import urllib.request as ureq
import pandas as pd

IBTRACS_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
BOM_WARNINGS_URL = "https://api.weather.bom.gov.au/v1/warnings"  # live only

AU_BBOX = ( -45.0, 110.0, -10.0, 160.0 )  # south, west, north, east

def within_bbox(lat, lon, bbox):
    s,w,n,e = bbox
    return (s <= lat <= n) and (w <= lon <= e)

def month_range(year:int, month:int):
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year+1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month+1, 1, tzinfo=timezone.utc)
    return start, end

def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print("Downloading:", url)
        with ureq.urlopen(url) as r, open(dest, "wb") as f:
            f.write(r.read())
    return dest

def harvest_ibtracs(year:int, month:int, bbox=AU_BBOX) -> pd.DataFrame:
    tmp = download(IBTRACS_URL, Path("data/_cache/ibtracs.csv"))
    df = pd.read_csv(tmp)
    # IBTrACS has one row per 6-hour timestep per storm with columns 'ISO_TIME','LAT','LON','USA_WIND','USA_PRES', etc.
    df = df.rename(columns=str.upper)
    df = df[["SID","NAME","ISO_TIME","LAT","LON","USA_WIND","USA_PRES"]].copy()
    df = df.dropna(subset=["ISO_TIME","LAT","LON"])
    # Parse times
    df["time"] = pd.to_datetime(df["ISO_TIME"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    start, end = month_range(year, month)
    df = df[(df["time"]>=start) & (df["time"]<end)]
    # Filter bbox (pad slightly to catch near-coast)
    s,w,n,e = bbox
    pad = 2.0
    df = df[(df["LAT"].between(s-pad, n+pad)) & (df["LON"].between(w-pad, e+pad))]
    # Label = 1 for any track point inside AU box, keep metadata
    keep = []
    for _, r in df.iterrows():
        lat, lon = float(r["LAT"]), float(r["LON"])
        if within_bbox(lat, lon, bbox):
            keep.append({
                "time": r["time"].strftime("%Y-%m-%dT%H:00:00Z"),
                "lat": lat, "lon": lon,
                "label": 1,
                "source": "IBTrACS",
                "details": f"{r['SID']}|{r.get('NAME','')}"
            })
    return pd.DataFrame(keep)

def harvest_bom_live(bbox=AU_BBOX) -> pd.DataFrame:
    # Live only: returns current warnings with geometry-less region names.
    # We approximate by tagging the whole AU box at current hour if any severe warning exists.
    try:
        with ureq.urlopen(BOM_WARNINGS_URL) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print("BoM live warnings fetch failed:", e)
        return pd.DataFrame(columns=["time","lat","lon","label","source","details"])
    items = data.get("warnings", []) if isinstance(data, dict) else []
    severe = [w for w in items if "Severe" in (w.get("headline","") or "") or "Thunderstorm" in (w.get("headline","") or "")]
    if not severe:
        return pd.DataFrame(columns=["time","lat","lon","label","source","details"])
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    # Tag a coarse grid of AU with label=1 (centroids every 2 degrees)
    s,w,n,e = bbox
    lats = [round(x,2) for x in frange(s, n, 2.0)]
    lons = [round(x,2) for x in frange(w, e, 2.0)]
    rows=[]
    for la in lats:
        for lo in lons:
            rows.append({
                "time": now.strftime("%Y-%m-%dT%H:00:00Z"),
                "lat": la, "lon": lo,
                "label": 1,
                "source": "BOM_live",
                "details": "active_severe_warning"
            })
    return pd.DataFrame(rows)

def frange(a,b,step):
    x=a
    while x<=b+1e-9:
        yield x
        x+=step

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--include-bom-live", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    Path("data").mkdir(parents=True, exist_ok=True)
    out = args.out or f"data/storm_labels_{args.year:04d}-{args.month:02d}.csv"

    ib = harvest_ibtracs(args.year, args.month)
    if args.include_bom_live:
        bl = harvest_bom_live()
        lab = pd.concat([ib, bl], ignore_index=True).drop_duplicates()
    else:
        lab = ib

    if not len(lab):
        print("No labels found for the given month.")
    lab.to_csv(out, index=False)
    print("Wrote", out, "rows:", len(lab))

if __name__ == "__main__":
    main()