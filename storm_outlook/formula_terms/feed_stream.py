"""FEED_FIZZLE_PREREG.md: ERA5 (ARCO) environmental terms at each event -> feed_terms/<k>.json. Resumable.  python feed_stream.py"""
import os, json, time, requests, numcodecs, numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter
B = "https://gcp-public-data-arco-era5.storage.googleapis.com/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/"
md = requests.get(B + ".zmetadata", timeout=60).json()["metadata"]; ses = requests.Session()
def get(v, key, shape):
    a = md[f"{v}/.zarray"]
    for k in range(5):
        try: r = ses.get(B + f"{v}/{key}", timeout=300); r.raise_for_status(); break
        except Exception:
            if k == 4: raise
            time.sleep(5 * (k + 1))
    return np.frombuffer(numcodecs.get_codec(a["compressor"]).decode(r.content), dtype=a["dtype"]).reshape(shape)
LEV = get("level", "0", (37,)).astype(float); IL = {p: list(LEV).index(p) for p in (200, 500, 600, 850, 925)}
tidx = lambda ts: int((pd.Timestamp(ts) - pd.Timestamp("1900-01-01")) / pd.Timedelta(hours=1))
lat = np.linspace(90, -90, 721); lon = np.arange(1440) * 0.25; a = 6.371e6; OM = 7.2921e-5; g = 9.81; Rd = 287.0
dl = np.deg2rad(0.25); dp = np.deg2rad(-0.25)
LSM = get("land_sea_mask", f"{tidx('2010-01-01')}.0.0", (721, 1440)).astype(float)
OUT = "feed_terms"; os.makedirs(OUT, exist_ok=True)
def box(la0, lo0, half=9.0):
    i = np.where(np.abs(lat - la0) <= half)[0]; w = min(180.0, half / max(np.cos(np.deg2rad(min(abs(la0) + half, 89.0))), 0.05))
    dlon = (lon - lo0 + 180) % 360 - 180; j = np.where(np.abs(dlon) <= w)[0]; return i, j[np.argsort(dlon[j])]
def gc(la1, lo1, la2, lo2):
    p1, p2 = np.deg2rad(la1), np.deg2rad(la2); return 6371.0 * np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(np.deg2rad(lo2 - lo1)), -1, 1))
def deriv(ub, vb, la):
    c = np.cos(np.deg2rad(la))[:, None]; m = a * np.maximum(c, 1e-3)
    zeta = (np.gradient(vb, dl, axis=-1) - np.gradient(ub * c, dp, axis=-2)) / m
    div = (np.gradient(ub, dl, axis=-1) + np.gradient(vb * c, dp, axis=-2)) / m; return zeta, div
def qsat(T, p):  # T K, p hPa -> kg/kg (Bolton)
    es = 6.112 * np.exp(17.67 * (T - 273.15) / (T - 29.65)); return 0.622 * es / (p - 0.378 * es)
def one(k, e, U, V, T, Q, SST):
    i, j = box(e.lat, e.lon); LA = lat[i][:, None] + 0 * lon[j][None]; LO = lon[j][None] + 0 * lat[i][:, None]; W = np.cos(np.deg2rad(LA))
    sg = 1.0 if e.lat >= 0 else -1.0; sub = lambda X, l: X[l][i][:, j].astype(np.float64)
    z850, d850 = deriv(sub(U, IL[850]), sub(V, IL[850]), lat[i]); zc = sg * z850
    zs = gaussian_filter(zc, 50 / 27.8, mode="nearest"); m0 = gc(e.lat, e.lon, LA, LO) <= 3 * 111.2
    q_ = np.argmax(np.where(m0, zs, -np.inf)); cla, clo = LA.flat[q_], LO.flat[q_]; d = gc(cla, clo, LA, LO)
    mean = lambda X, msk: float(np.sum(X[msk] * W[msk]) / np.sum(W[msk]))
    core, ring = d <= 75, (d >= 200) & (d <= 800); f = 2 * OM * abs(np.sin(np.deg2rad(cla)))
    zstar = mean(zc, core); Ro = zstar / (2 * f)
    Vr = {p: np.array([mean(sub(U, IL[p]), ring), mean(sub(V, IL[p]), ring)]) for p in (200, 500, 850)}
    S = float(np.hypot(*(Vr[200] - Vr[850]))); S_low = float(np.hypot(*(Vr[500] - Vr[850])))
    T850, T500 = mean(sub(T, IL[850]), ring), mean(sub(T, IL[500]), ring); th = lambda Tk, p: Tk * (1000 / p) ** 0.2857
    dz = Rd / g * 0.5 * (T850 + T500) * np.log(850 / 500); N2 = g * np.log(th(T500, 500) / th(T850, 850)) / dz
    sigE = float(0.31 * f * S_low / (np.sqrt(max(N2, 1e-6)) * dz))
    T600, Q600 = sub(T, IL[600]), sub(Q, IL[600]); RH600 = mean(np.clip(Q600 / qsat(T600, 600.0), 0, 1.2), ring)
    sst = SST[i][:, j].astype(np.float64); lsm = LSM[i][:, j]; oc = (d <= 400) & (lsm < 0.5) & np.isfinite(sst) & (sst > 250)
    dq = mean((qsat(sst, 1000.0) - sub(Q, IL[925])) * 1000, oc) if oc.sum() >= 5 else 0.0
    land = mean(lsm, d <= 300)
    _, d200 = deriv(sub(U, IL[200]), sub(V, IL[200]), lat[i]); div200 = mean(d200, d <= 400)
    json.dump(dict(k=int(k), clat=float(cla), clon=float(clo), f=f, zstar=zstar, Ro=Ro, S=S, S_low=S_low, N2=float(N2), sigE=sigE, RH600=RH600, dq=float(dq), land=land, div200=div200), open(os.path.join(OUT, f"{k}.json"), "w"))
if __name__ == "__main__":
    EV = pd.read_csv("feed_events.csv", parse_dates=["t"]); EV["k"] = EV.index; t0 = time.time(); n = 0; bad = []
    for h, sub_ in EV.groupby("t"):
        sub_ = sub_[[not os.path.exists(os.path.join(OUT, f"{k}.json")) for k in sub_.k]]
        if not len(sub_): continue
        try:
            ti = tidx(h); key = f"{ti}.0.0.0"; sh = (37, 721, 1440)
            U = get("u_component_of_wind", key, sh); V = get("v_component_of_wind", key, sh); T = get("temperature", key, sh); Q = get("specific_humidity", key, sh)
            SST = get("sea_surface_temperature", f"{ti}.0.0", (721, 1440))
            for _, e in sub_.iterrows():
                try: one(e.k, e, U, V, T, Q, SST)
                except Exception as ex: bad.append((int(e.k), type(ex).__name__, str(ex)[:60]))
            del U, V, T, Q, SST
        except Exception as ex: bad.append((str(h), type(ex).__name__, str(ex)[:80]))
        n += 1
        if n % 10 == 0: print(n, round(time.time() - t0), "s", flush=True)
    print("done", n, "failed", len(bad), bad[:8], round(time.time() - t0), "s", flush=True)
