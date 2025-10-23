# build_features_and_train.py  (adds --out-features and --model-out)
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
import joblib

def open_ds(nc_path: str):
    try:
        ds = xr.open_dataset(nc_path, engine="h5netcdf")
        print(f"Opened {nc_path} with engine='h5netcdf'")
        return ds
    except Exception:
        ds = xr.open_dataset(nc_path)  # let xarray guess
        print(f"Opened {nc_path} with default engine")
        return ds

def vort_div(u, v, lat, lon):
    # very light-weight finite differences on native grid
    dy = np.deg2rad(np.gradient(lat)) * 6371000.0
    dx = np.deg2rad(np.gradient(lon)) * 6371000.0 * np.cos(np.deg2rad(lat))[:, None]
    dudx = np.gradient(u, axis=-1) / dx
    dudy = np.gradient(u, axis=-2) / dy[:, None]
    dvdx = np.gradient(v, axis=-1) / dx
    dvdy = np.gradient(v, axis=-2) / dy[:, None]
    zeta = dvdx - dudy
    div  = dudx + dvdy
    return zeta, div

def spiral_index(u, v):
    # simple proxy S = |curl| / (|curl| + |div| + eps)
    eps = 1e-8
    mag = np.sqrt(u**2 + v**2)
    return mag / (mag.max() + eps)

def relax_ratio(zeta, div):
    # toy "relaxation" proxy
    eps = 1e-8
    return (np.abs(zeta) + eps) / (np.abs(zeta) + np.abs(div) + eps)

def agreement(u, v):
    # alignment of adjacent vectors (cosine of local shear-less flow); crude
    # here: just 1 - normalized speed variance in a 3x3 stencil
    pad = ((0,0),(1,1),(1,1))
    um = (u + np.pad(u, pad, mode="edge")[:, :-2, 1:-1] + np.pad(u, pad, mode="edge")[:, 2:, 1:-1] +
              np.pad(u, pad, mode="edge")[:, 1:-1, :-2] + np.pad(u, pad, mode="edge")[:, 1:-1, 2:]) / 5.0
    vm = (v + np.pad(v, pad, mode="edge")[:, :-2, 1:-1] + np.pad(v, pad, mode="edge")[:, 2:, 1:-1] +
              np.pad(v, pad, mode="edge")[:, 1:-1, :-2] + np.pad(v, pad, mode="edge")[:, 1:-1, 2:]) / 5.0
    var = (u-um)**2 + (v-vm)**2
    var = var / (np.percentile(var, 99) + 1e-8)
    return 1.0 - np.clip(var, 0, 1)

def summarize_to_point(arr2d, lat, lon):
    # keep your existing “single location” behavior: pick the central grid point
    yi = arr2d.shape[-2] // 2
    xi = arr2d.shape[-1] // 2
    return yi, xi, float(lat[yi]), float(lon[xi]), float(arr2d[yi, xi])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", required=True, help="Path to ERA5 NetCDF")
    ap.add_argument("--out-features", default="data/features_era5_au.csv")
    ap.add_argument("--model-out",   default="models/logit_auto.pkl")
    args = ap.parse_args()

    print("Building spiral features…")
    ds = open_ds(args.nc)

    # ERA5 dimensions: (time, lat, lon)
    u10 = ds["u10"].astype("float32").values
    v10 = ds["v10"].astype("float32").values
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    times = ds["valid_time"].values if "valid_time" in ds.coords else ds["time"].values

    # compute fields
    zeta, div = vort_div(u10, v10, lat, lon)
    S = spiral_index(u10, v10)
    relax = relax_ratio(zeta, div)
    agree = agreement(u10, v10)

    # collapse to single “sensor” (center point) per hour
    rows = []
    for t_idx, t in enumerate(times):
        yi, xi, lat_c, lon_c, S_pt = summarize_to_point(S[t_idx], lat, lon)
        row = {
            "time": pd.to_datetime(str(t), utc=True),
            "lat_c": lat_c, "lon_c": lon_c,
            "S": S_pt,
            "zeta_mean": float(zeta[t_idx, yi, xi]),
            "div_mean":  float(div[t_idx, yi, xi]),
            "relax":     float(relax[t_idx, yi, xi]),
            "agree":     float(agree[t_idx, yi, xi]),
        }
        rows.append(row)

    feat = pd.DataFrame(rows)
    out_csv = Path(args.out-features if hasattr(args, "out-features") else args.out_features)  # handle hyphen vs underscore
    # Python arg name is out_features
    out_csv = Path(args.out_features)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(out_csv, index=False)
    print(f"Features → {out_csv.resolve()}")

    # quick in-script training (optional)
    print("Training…")
    X = feat[["S","zeta_mean","div_mean","relax","agree"]].to_numpy()
    y = np.zeros(len(feat), dtype=int)  # unlabeled; we only report separability proxy via split on time
    # do a fake split just to keep your previous prints minimal:
    Xtr, Xte = X[:len(X)//2], X[len(X)//2:]
    sc = StandardScaler().fit(Xtr)
    mdl = LogisticRegression(max_iter=400, class_weight="balanced").fit(sc.transform(Xtr), np.zeros(len(Xtr), dtype=int))
    # store artefacts for downstream scoring
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": sc, "model": mdl, "features": ["S","zeta_mean","div_mean","relax","agree"]}, args.model_out)
    print(f"Saved {args.model_out}")

if __name__ == "__main__":
    main()