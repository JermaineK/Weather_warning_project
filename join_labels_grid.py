#!/usr/bin/env python3
# Join gridded ERA5 feature snapshots with storm labels (IBTrACS/besttrack)
# to produce three targets per grid cell and time: storm, near_storm, pregen.

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------- I/O helpers ----------------

def _read_any(path: str) -> pd.DataFrame:
    low = path.lower()
    if low.endswith((".parquet", ".parq", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)

def _write_any(path: str, df: pd.DataFrame) -> None:
    low = path.lower()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if low.endswith((".parquet", ".parq", ".pq")):
        df.to_parquet(path, index=False)
    else:
        comp = "gzip" if low.endswith(".gz") else "infer"
        df.to_csv(path, index=False, compression=comp)

# ---------------- args & small utils ----------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Join gridded features with storm labels (IBTrACS/besttrack)."
    )
    ap.add_argument("--features", required=True, help="CSV/GZ or Parquet of gridded features (has time,lat,lon).")
    ap.add_argument("--labels",   required=True, help="CSV/GZ or Parquet of labels (has time,lat,lon[,name]).")
    ap.add_argument("--out",      required=True, help="Output CSV/GZ or Parquet path.")

    # matching radii (degrees) and time windows (hours)
    ap.add_argument("--storm-radius-deg",  type=float, default=1.0,  help="Tight radius for 'storm'.")
    ap.add_argument("--storm-time-h",      type=float, default=3.0,  help="± hours around feature time for 'storm'.")
    ap.add_argument("--near-radius-deg",   type=float, default=5.0,  help="Looser radius for 'near_storm'.")
    ap.add_argument("--near-time-h",       type=float, default=12.0, help="± hours around feature time for 'near_storm'.")
    ap.add_argument("--pregen-radius-deg", type=float, default=5.0,  help="Radius for 'pregen' (future window).")
    ap.add_argument("--pregen-future-h",   type=float, default=48.0, help="Future horizon (hours) for 'pregen'.")

    # quality/perf options
    ap.add_argument("--normalize-lon", choices=["none","-180..180","0..360"], default="none",
                    help="Optional longitude normalization applied to BOTH inputs.")
    ap.add_argument("--labels-time-pad-h", type=float, default=0.0,
                    help="Extra time padding on label window (helps if label hours are slightly off).")
    ap.add_argument("--track-time-offset-hours", type=float, default=0.0,
                    help="Shift ALL label timestamps by this many hours (e.g., +2).")
    ap.add_argument("--area", default=None,
                    help="Optional crop 'latN,lonW,latS,lonE' applied AFTER lon normalization.")
    return ap.parse_args()

def _norm_lon(a: pd.Series, mode: str) -> pd.Series:
    if mode == "none":
        return pd.to_numeric(a, errors="coerce")
    x = pd.to_numeric(a, errors="coerce").astype(float)
    if mode == "0..360":
        x = (x % 360 + 360) % 360
    else:
        # -180..180
        x = ((x + 180) % 360) - 180
    return x

def _parse_area(area: Optional[str]) -> Optional[Tuple[float,float,float,float]]:
    if not area:
        return None
    try:
        n, w, s, e = map(float, [p.strip() for p in area.split(",")])
        return (n, w, s, e)
    except Exception:
        raise ValueError("--area must be 'latN,lonW,latS,lonE'")

def _crop_aoi(df: pd.DataFrame, aoi: Tuple[float,float,float,float]) -> pd.DataFrame:
    n, w, s, e = aoi
    # latitude
    df = df[(df["lat"] <= n) & (df["lat"] >= s)]
    # longitude (handle wrap)
    if w <= e:
        return df[(df["lon"] >= w) & (df["lon"] <= e)]
    else:
        # wrap across anti-meridian: (lon ≥ w) or (lon ≤ e)
        return df[(df["lon"] >= w) | (df["lon"] <= e)]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.deg2rad(lat2 - lat1)
    dlon = np.deg2rad(lon2 - lon1)
    a = np.sin(dlat/2.0)**2 + np.cos(np.deg2rad(lat1))*np.cos(np.deg2rad(lat2))*np.sin(dlon/2.0)**2
    return 2.0 * R * np.arcsin(np.sqrt(a))

def match_mask(df: pd.DataFrame,
               lab_by_hour: Dict[pd.Timestamp, pd.DataFrame],
               radius_deg: float,
               time_h: float,
               future_only: bool,
               time_pad_h: float = 0.0) -> np.ndarray:
    """
    For each time slice of df (by 'time_hr'), mark grid cells with ≥1 label
    within 'radius_deg' and within the time window.
    If future_only=True: (t, t+dt]; else: [t-dt, t+dt].
    lab_by_hour is a dict: {timestamp_hour: label_subframe}.
    Returns int array of shape (len(df),) with 0/1 flags.
    """
    out = np.zeros(len(df), dtype=int)
    dt = pd.Timedelta(hours=time_h + time_pad_h)

    lab_hours = sorted(lab_by_hour.keys())
    if not lab_hours:
        return out

    for t, chunk in df.groupby("time_hr", sort=True):
        if chunk.empty:
            continue

        if future_only:
            # (t, t+dt] → exclude t itself
            t_min = t + pd.Timedelta(nanoseconds=1)
            t_max = t + dt
        else:
            # [t-dt, t+dt]
            t_min = t - dt
            t_max = t + dt

        cand = []
        for th in lab_hours:
            if th < t_min or th > t_max:
                continue
            g = lab_by_hour.get(th)
            if g is not None and not g.empty:
                cand.append(g)
        if not cand:
            continue
        sel = pd.concat(cand, ignore_index=True)

        lat1 = chunk["lat"].to_numpy()
        lon1 = chunk["lon"].to_numpy()
        lat2 = sel["lat"].to_numpy()
        lon2 = sel["lon"].to_numpy()

        dkm = haversine_km(lat1[:, None], lon1[:, None], lat2[None, :], lon2[None, :])
        hit = (dkm <= (111.0 * radius_deg)).any(axis=1)

        out[chunk.index.values] = hit.astype(int)

    return out

# ---------------- main ----------------

def main():
    args = parse_args()

    # Read inputs (any format), basic checks
    df = _read_any(args.features)
    lb = _read_any(args.labels)

    for need in ("time","lat","lon"):
        if need not in df.columns:
            raise ValueError(f"{args.features}: missing column '{need}'")
        if need not in lb.columns:
            raise ValueError(f"{args.labels}: missing column '{need}'")

    # Parse & coerce
    for c in ("lat","lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        lb[c] = pd.to_numeric(lb[c], errors="coerce")

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    lb["time"] = pd.to_datetime(lb["time"], utc=True, errors="coerce").dt.tz_localize(None)

    # Optional label time shift
    if args.track_time_offset_hours != 0.0:
        lb["time"] = lb["time"] + pd.Timedelta(hours=float(args.track_time_offset_hours))

    # Drop impossible rows
    df = df.dropna(subset=["time","lat","lon"]).reset_index(drop=True)
    lb = lb.dropna(subset=["time","lat","lon"]).reset_index(drop=True)

    # Optional lon normalization (both) — keeps globe; no hidden regional filter
    if args.normalize_lon != "none":
        df["lon"] = _norm_lon(df["lon"], args.normalize_lon)
        lb["lon"] = _norm_lon(lb["lon"], args.normalize_lon)

    # Optional AOI crop (after lon normalization)
    aoi = _parse_area(args.area)
    if aoi is not None:
        df = _crop_aoi(df, aoi)
        lb = _crop_aoi(lb, aoi)

    # Quick ranges printout
    def _rng(s: pd.Series) -> str:
        return f"{s.min():.3f}..{s.max():.3f}"
    if not df.empty and not lb.empty:
        print(f"[join] FEATURES time: {df['time'].min()} → {df['time'].max()}  "
              f"lat: { _rng(df['lat']) }  lon({args.normalize_lon}): { _rng(df['lon']) }")
        print(f"[join] LABELS   time: {lb['time'].min()} → {lb['time'].max()}  "
              f"lat: { _rng(lb['lat']) }  lon({args.normalize_lon}): { _rng(lb['lon']) }")

    # hourly bins
    df["time_hr"] = df["time"].dt.floor("h")
    lb["time_hr"] = lb["time"].dt.floor("h")

    # build label dict by hour (only lat/lon/time_hr kept)
    lab_by_hour: Dict[pd.Timestamp, pd.DataFrame] = {
        t: g[["time_hr","lat","lon"]].copy()
        for t, g in lb.groupby("time_hr", sort=True)
    }

    # targets
    df["storm"]      = match_mask(df, lab_by_hour, args.storm_radius_deg,  args.storm_time_h,     future_only=False, time_pad_h=args.labels_time_pad_h)
    df["near_storm"] = match_mask(df, lab_by_hour, args.near_radius_deg,   args.near_time_h,      future_only=False, time_pad_h=args.labels_time_pad_h)
    df["pregen"]     = match_mask(df, lab_by_hour, args.pregen_radius_deg, args.pregen_future_h,  future_only=True,  time_pad_h=args.labels_time_pad_h)

    # write (any format)
    out = args.out
    _write_any(out, df.drop(columns=["time_hr"], errors="ignore"))

    print(
        f"[join] Wrote {out}  rows={len(df):,} | "
        f"Pos (storm/near/pregen): {int(df['storm'].sum()):,} / {int(df['near_storm'].sum()):,} / {int(df['pregen'].sum()):,}"
    )

if __name__ == "__main__":
    main()