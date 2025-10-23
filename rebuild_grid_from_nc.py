#!/usr/bin/env python
import argparse, numpy as np, pandas as pd, xarray as xr
from pathlib import Path

def _coord_name(ds, candidates):
    for c in candidates:
        if c in ds.coords or c in ds.dims:
            return c
    raise ValueError(f"None of {candidates} found. Dims={tuple(ds.dims)}, Coords={tuple(ds.coords)}")

def _data_name(ds, candidates):
    for c in candidates:
        if c in ds:
            return c
    return None

def compute_zeta_div(u3, v3, lat, lon):
    """
    u3, v3: arrays shaped (T, Y, X) in m/s
    lat, lon: 1D arrays (degrees), length Y and X
    Returns (zeta, div) both (T, Y, X) in s^-1
    """
    u3 = np.asarray(u3, dtype="float64")
    v3 = np.asarray(v3, dtype="float64")

    # meters per degree (approx)
    m_per_deg_y = 110_540.0
    cosphi = np.cos(np.deg2rad(lat))             # (Y,)
    m_per_deg_x = 111_320.0 * cosphi            # (Y,)

    # coordinate metrics in meters
    dlat_deg = np.gradient(lat)                  # (Y,)
    dlon_deg = np.gradient(lon)                  # (X,)
    y_m = np.cumsum(np.r_[0.0, m_per_deg_y * dlat_deg[1:]])     # (Y,)
    # For x we’ll start with a nominal meter axis at reference latitude,
    # then correct derivatives with a latitude-dependent scale.
    x_m_nominal = np.cumsum(np.r_[0.0, (m_per_deg_x[0] * dlon_deg[1:])])  # (X,)

    T, Y, X = u3.shape
    zeta = np.empty_like(u3)
    div  = np.empty_like(u3)

    # 2-D longitude scale to apply to ∂/∂x terms: shape (Y,1) -> broadcasts to (Y,X)
    scale2d = (m_per_deg_x / m_per_deg_x[0])[:, None]  # (Y,1)

    for t in range(T):
        # basic Cartesian gradients on (y_m, x_m_nominal)
        dU_dy, dU_dx = np.gradient(u3[t], y_m, x_m_nominal, edge_order=1)  # each (Y,X)
        dV_dy, dV_dx = np.gradient(v3[t], y_m, x_m_nominal, edge_order=1)

        # correct x-derivatives for latitude-varying meters/deg
        dU_dx = dU_dx * scale2d
        dV_dx = dV_dx * scale2d

        zeta[t] = dV_dx - dU_dy
        div[t]  = dU_dx + dV_dy

    return zeta, div

def main(nc_path: Path, out_csv: Path):
    print(f"Opening {nc_path} …")
    ds = xr.open_dataset(nc_path, engine="h5netcdf")

    # coord + var names
    tname = _coord_name(ds, ["time", "valid_time"])
    yname = _coord_name(ds, ["latitude", "lat"])
    xname = _coord_name(ds, ["longitude", "lon"])

    u_name = _data_name(ds, ["u10", "u"])
    v_name = _data_name(ds, ["v10", "v"])
    if u_name is None or v_name is None:
        raise SystemExit(f"Need u- and v-wind (u10/v10 or u/v). Found: {list(ds.data_vars)}")
    p_name = _data_name(ds, ["msl", "mslp", "sp", "pres", "pressure"])

    # pull arrays as (T,Y,X)
    u = ds[u_name].transpose(tname, yname, xname).astype("float64").values
    v = ds[v_name].transpose(tname, yname, xname).astype("float64").values
    lats = ds[yname].values
    lons = ds[xname].values

    flipped = False
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        u = u[:, ::-1, :]
        v = v[:, ::-1, :]
        flipped = True

    print("Computing zeta/div from winds …")
    zeta, div = compute_zeta_div(u, v, lats, lons)

    # flatten to table
    time_vals = ds[tname].values
    TT, YY, XX = np.meshgrid(time_vals, lats, lons, indexing="ij")

    out = {
        "time": TT.ravel(),
        "lat":  YY.ravel(),
        "lon":  XX.ravel(),
        "u10":  u.ravel(),
        "v10":  v.ravel(),
        "zeta": zeta.ravel(),
        "div":  div.ravel(),
    }

    if p_name:
        p = ds[p_name].transpose(tname, yname, xname).values
        if flipped:
            p = p[:, ::-1, :]
        # Pa→hPa if obviously in Pa
        if np.nanmedian(p) > 2_000:
            p = p / 100.0
        out["msl"] = p.ravel()

    df = pd.DataFrame(out)
    df["S"]     = np.hypot(df["zeta"], df["div"])
    df["agree"] = (np.abs(df["zeta"]) > np.abs(df["div"])).astype(float)

    df = df.sort_values(["time","lat","lon"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    comp = "gzip" if out_csv.suffix.endswith(".gz") else None
    df.to_csv(out_csv, index=False, compression=comp)
    print(f"Wrote {out_csv} rows: {len(df):,}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc",  required=True, help="Path to ERA5 NetCDF (.nc)")
    ap.add_argument("--out", required=True, help="Output CSV(.gz)")
    args = ap.parse_args()
    main(Path(args.nc), Path(args.out))