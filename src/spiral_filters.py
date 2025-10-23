import numpy as np

def spiral_index(u: np.ndarray, v: np.ndarray) -> float:
    phi = np.arctan2(v, u)
    dphix = np.diff(np.unwrap(phi, axis=1), axis=1)
    dphiy = np.diff(np.unwrap(phi, axis=0), axis=0)
    pad_x = np.pad(dphix, ((0,0),(1,0)), mode='edge')
    pad_y = np.pad(dphiy, ((1,0),(0,0)), mode='edge')
    dphi = 0.5*(pad_x + pad_y)
    w = np.hypot(u, v) + 1e-9
    return float(np.sum(np.sign(dphi) * w) / np.sum(w))

def vorticity_divergence(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    Re = 6.371e6
    lat_r = np.deg2rad(lat)
    dlat = np.deg2rad(np.gradient(lat))
    dlon = np.deg2rad(np.gradient(lon))
    phi = lat_r[:, None]
    cosphi = np.cos(phi)
    dy = Re * dlat[:, None]
    dx = Re * (dlon[None, :]) * cosphi
    du_dy = np.gradient(u, axis=0) / (dy + 1e-12)
    dv_dx = np.gradient(v, axis=1) / (dx + 1e-12)
    dv_dy = np.gradient(v, axis=0) / (dy + 1e-12)
    du_dx = np.gradient(u, axis=1) / (dx + 1e-12)
    zeta = dv_dx - du_dy
    div  = du_dx + dv_dy
    return zeta, div

def relaxation_ratio(zeta: np.ndarray) -> float:
    z = np.abs(zeta)
    z_med = np.nanmedian(z) + 1e-12
    return float(np.nanmean((z / z_med)**2))

def multi_scale_stability(S2: float, S3: float, S5: float) -> int:
    s2, s3, s5 = np.sign(S2), np.sign(S3), np.sign(S5)
    return int((s2 == s3) and (s3 == s5))

def parity_agreement(S: float, zeta_mean: float) -> int:
    return int(np.sign(S) == np.sign(zeta_mean))
