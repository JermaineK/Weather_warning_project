# score_nowcast.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import joblib

from features import open_ds, spiral_index, vort_div, relax_ratio, mobility_metrics

def build_features(nc_path: str, north=-10.0, south=-25.0, west=135.0, east=155.0) -> pd.DataFrame:
    ds = open_ds(nc_path)
    tcoord = next((c for c in ("time","valid_time","forecast_time") if c in ds.coords), None)
    if tcoord is None: raise RuntimeError("No time coordinate found.")
    lats = ds["latitude"].values; lons = ds["longitude"].values
    ilat = (lats < north) & (lats > south)
    ilon = (lons > west)  & (lons < east)
    lat_crop, lon_crop = lats[ilat], lons[ilon]
    times = ds[tcoord].values[:]

    rows=[]
    for t in times:
        u = ds["u10"].sel({tcoord: t}).values[ilat][:, ilon]
        v = ds["v10"].sel({tcoord: t}).values[ilat][:, ilon]
        S = spiral_index(u, v)
        zeta, div = vort_div(u, v, lat_crop, lon_crop)
        zmean = float(np.nanmean(zeta))
        relax = relax_ratio(zeta)
        agree = int(np.sign(S) == np.sign(zmean))
        mob = mobility_metrics(u, v, zeta, div)
        rows.append({
            "time": str(np.datetime_as_string(t, timezone="UTC")),
            "lat_c": float(np.mean(lat_crop)), "lon_c": float(np.mean(lon_crop)),
            "S": S, "zeta_mean": zmean, "div_mean": float(np.nanmean(div)),
            "relax": relax, "agree": agree,
            **mob
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", required=True, help="ERA5 NetCDF path")
    ap.add_argument("--model", default="models/logit_era5_au.pkl", help="joblib file (logit)")
    ap.add_argument("--out", default="results/nowcast_scores.csv")
    ap.add_argument("--north", type=float, default=-10.0)
    ap.add_argument("--south", type=float, default=-25.0)
    ap.add_argument("--west",  type=float, default=135.0)
    ap.add_argument("--east",  type=float, default=155.0)
    args = ap.parse_args()

    Path("results").mkdir(parents=True, exist_ok=True)
    feat = build_features(args.nc, args.north, args.south, args.west, args.east)

    # Use the trained logistic model; we saved {"scaler":..., "model":...}
    blob = joblib.load(args.model)
    scaler: StandardScaler = blob["scaler"]
    model: LogisticRegression = blob["model"]

    X = feat[["S","zeta_mean","div_mean","relax","agree"]].astype(float).to_numpy()
    Xs = scaler.transform(X)
    feat["risk"] = model.predict_proba(Xs)[:,1]

    # simple threshold: focus on high-risk coherent spirals
    feat["flag"] = (feat["risk"] >= 0.5).astype(int)

    feat.to_csv(args.out, index=False)
    print("Wrote", args.out, "rows:", len(feat), "high-risk:", int(feat["flag"].sum()))

if __name__ == "__main__":
    main()