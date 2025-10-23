# join_labels_to_features.py  (tolerant time match)
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features_era5_au.csv")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--radius_deg", type=float, default=1.0, help="spatial match radius in degrees (~100 km)")
    ap.add_argument("--time_tol_hours", type=int, default=3, help="time tolerance (± hours) when matching labels")
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    lb = pd.read_csv(args.labels)

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    lb["time"] = pd.to_datetime(lb["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time","lat_c","lon_c"]).reset_index(drop=True)
    lb = lb.dropna(subset=["time","lat","lon"]).reset_index(drop=True)

    # Round to hour and pre-bucket labels by hour to speed lookups
    df["time_hr"] = df["time"].dt.floor("H")
    lb["time_hr"] = lb["time"].dt.floor("H")

    # Build a dict of label indices per hour within tolerance window
    # For each unique feature hour, collect labels from [t-±tol]
    tol = pd.Timedelta(hours=args.time_tol_hours)
    unique_hours = df["time_hr"].sort_values().unique()
    lb_by_time = lb.set_index("time_hr").sort_index()

    out_chunks = []
    for t in unique_hours:
        # pull labels in window
        win = lb_by_time.loc[t - tol : t + tol]
        block = df[df["time_hr"] == t].copy()
        if win.empty:
            block["storm"] = 0
            out_chunks.append(block)
            continue

        lat2 = win["lat"].to_numpy()
        lon2 = win["lon"].to_numpy()

        lat1 = block["lat_c"].to_numpy()
        lon1 = block["lon_c"].to_numpy()

        storm = np.zeros(len(block), dtype=int)
        for i in range(len(block)):
            dlat = np.abs(lat2 - lat1[i])
            dlon = np.abs(lon2 - lon1[i])
            # quick degree-box test (fast, fine at 0.25° grids)
            mask = (dlat <= args.radius_deg) & (dlon <= args.radius_deg)
            if mask.any():
                storm[i] = 1
        block["storm"] = storm
        out_chunks.append(block)

    labdf = pd.concat(out_chunks, ignore_index=True)
    labdf["storm"] = labdf["storm"].astype(int)

    # Clean extras if present
    for c in ["lat","lon","label","source","details","time_hr"]:
        if c in labdf.columns:
            labdf = labdf.drop(columns=c)

    outp = args.out or args.features.replace(".csv", "_labelled.csv")
    Path(outp).parent.mkdir(parents=True, exist_ok=True)
    labdf.to_csv(outp, index=False)
    print(f"Wrote {outp} rows: {len(labdf)}  positives: {int(labdf['storm'].sum())}")

if __name__ == "__main__":
    main()