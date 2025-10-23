# features.py
from __future__ import annotations
import numpy as np
import xarray as xr

def open_ds(nc_path: str):
    for eng in ("h5netcdf", "netcdf4", "scipy"):
        try:
            ds = xr.open_dataset(nc_path, engine=eng)
            print(f"Opened {nc_path} with engine='{eng}'")
            return ds
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not open {nc_path}. Last error: {last_err}")

def spiral_index(u: np.ndarray, v: np.ndarray) -> float:
    phi = np.arctan2(v, u)
    dphix = np.diff(np.unwrap(phi, axis=1), axis=1)
    dphiy = np.diff(np.unwrap(phi, axis=0), axis=0)
    pad_x = np.pad(dphix, ((0,0),(1,0)), mode='edge')
    pad_y = np.pad(dphiy, ((1,0),(0,0)), mode='edge')
    dphi = 0.5*(pad_x + pad_y)
    w = np.hypot(u, v) + 1e-9
    return float(np.sum(np.sign(dphi) * w) / np.sum(w))

def vort_div(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    Re = 6.371e6
    dlat = np.deg2rad(np.gradient(lat))
    dlon = np.deg2rad(np.gradient(lon))
    phi = np.deg2rad(lat)[:, None]
    dy = Re * dlat[:, None]
    dx = Re * (dlon[None, :]) * np.cos(phi)
    du_dy = np.gradient(u, axis=0) / (dy + 1e-12)
    dv_dx = np.gradient(v, axis=1) / (dx + 1e-12)
    dv_dy = np.gradient(v, axis=0) / (dy + 1e-12)
    du_dx = np.gradient(u, axis=1) / (dx + 1e-12)
    zeta = dv_dx - du_dy
    div  = du_dx + dv_dy
    return zeta, div

def relax_ratio(zeta: np.ndarray) -> float:
    z = np.abs(zeta)
    z_med = np.nanmedian(z) + 1e-12
    return float(np.nanmean((z / z_med)**2))

def mobility_metrics(u: np.ndarray, v: np.ndarray, zeta: np.ndarray, div: np.ndarray) -> dict:
    # “Freedom of movement” proxy suite
    ws = np.hypot(u, v)
    dir_rad = np.arctan2(v, u)
    # circular variance (directional spread)
    sinm, cosm = np.nanmean(np.sin(dir_rad)), np.nanmean(np.cos(dir_rad))
    R = np.hypot(sinm, cosm)
    circ_var = 1.0 - R
    return {
        "ws_mean": float(np.nanmean(ws)),
        "ws_var":  float(np.nanvar(ws)),
        "dir_var": float(circ_var),
        "shear_var": float(np.nanvar(zeta)),
        "div_var": float(np.nanvar(div)),
    }