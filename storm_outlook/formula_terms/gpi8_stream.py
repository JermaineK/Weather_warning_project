"""GENESIS_SCALE_PREREG.md (gpi_stream.py code, fresh days, output gpi8_days): ERA5 00 UTC per-point eta850, RH600, PI, 850-200 shear (300 km disc means). No labels.
Writes gpi_days/<YYYY-MM-DD>.npz. Resumable.  python gpi_stream.py <part> <nparts>"""
import os, sys, time, numpy as np, pandas as pd, scipy.sparse as sp
from tcpyPI import pi as bepi
_s = open("feed_stream.py", encoding="utf8").read(); exec(_s[:_s.find("def one(")])     # get, tidx, LEV, IL, lat, lon, LSM, a, OM, dl, dp, gc, qsat
OUT = "gpi8_days"; os.makedirs(OUT, exist_ok=True)
I0, I1 = int((90 - 40) / 0.25), int((90 + 40) / 0.25) + 1; BL = lat[I0:I1]; NB = (I1 - I0) * 1440
PLAT = np.r_[-np.arange(35, 2.4, -2.5), np.arange(2.5, 35.1, 2.5)]; PLON = np.arange(0, 360, 2.5)
PL = np.repeat(PLAT, len(PLON)); PO = np.tile(PLON, len(PLAT))
def build_W():
    rows, cols, w, wo = [], [], [], []
    lsm = LSM[I0:I1]
    for q, (la0, lo0) in enumerate(zip(PL, PO)):
        i = np.where(np.abs(BL - la0) <= 2.8)[0]; dlo = (lon - lo0 + 180) % 360 - 180; j = np.where(np.abs(dlo) <= 2.8 / np.cos(np.deg2rad(abs(la0) + 2.8)))[0]
        II, JJ = np.meshgrid(i, j, indexing="ij"); m = gc(la0, lo0, BL[II], lon[JJ]) <= 300.0; II, JJ = II[m], JJ[m]
        cw = np.cos(np.deg2rad(BL[II])); rows += [q] * len(II); cols += list(II * 1440 + JJ); w += list(cw); wo += list(cw * (lsm[II, JJ] < 0.5))
    W = sp.csr_matrix((w, (rows, cols)), shape=(len(PL), NB)); Wo = sp.csr_matrix((wo, (rows, cols)), shape=(len(PL), NB))
    s = np.asarray(W.sum(1)).ravel(); so = np.asarray(Wo.sum(1)).ravel(); no = np.asarray((Wo > 0).sum(1)).ravel()
    land = 1 - so / s
    return sp.diags(1 / s) @ W, sp.diags(1 / np.maximum(so, 1e-12)) @ Wo, land, no
W, WO, LANDF, NOC = build_W()
PT_LSM = LSM[((90 - PL) / 0.25).astype(int), (PO / 0.25).astype(int)]
KEEP = (LANDF < 0.5) & (PT_LSM < 0.5) & (NOC >= 5)
def band(X): return X[..., I0:I1, :].reshape(X.shape[:-2] + (NB,)).astype(np.float64)
def one(day):
    fo = os.path.join(OUT, f"{day:%Y-%m-%d}.npz")
    if os.path.exists(fo): return
    ti = tidx(day); key = f"{ti}.0.0.0"; sh = (37, 721, 1440); k2 = f"{ti}.0.0"
    U = get("u_component_of_wind", key, sh); V = get("v_component_of_wind", key, sh)
    u8, v8 = U[IL[850], I0 - 1:I1 + 1].astype(np.float64), V[IL[850], I0 - 1:I1 + 1].astype(np.float64)
    la = lat[I0 - 1:I1 + 1]; c = np.cos(np.deg2rad(la))[:, None]
    dvdx = (np.roll(v8, -1, 1) - np.roll(v8, 1, 1)) / (2 * dl); duc = (u8[:-2] * c[:-2] - u8[2:] * c[2:]) / (2 * (-dp))
    zeta = (dvdx[1:-1] - duc) / (a * c[1:-1]); eta = zeta + 2 * OM * np.sin(np.deg2rad(BL))[:, None]
    du = band(U[IL[200]]) - band(U[IL[850]]); dv = band(V[IL[200]]) - band(V[IL[850]]); del U, V
    T = get("temperature", key, sh); Q = get("specific_humidity", key, sh)
    Tb = band(T); Qb = band(Q); del T, Q
    SST = get("sea_surface_temperature", k2, (721, 1440)).astype(np.float64); MSL = get("mean_sea_level_pressure", k2, (721, 1440)).astype(np.float64)
    sstb = SST[I0:I1].ravel(); okb = np.isfinite(sstb) & (sstb > 250); sstb = np.where(okb, sstb, 0.0)
    ETA = np.abs(W @ eta.ravel()); VS = np.hypot(W @ du, W @ dv)
    Tm = (W @ Tb.T).T; Qm = (W @ Qb.T).T                                              # (37, npts)
    MS = W @ MSL[I0:I1].ravel() / 100.0
    wsum = WO @ okb.astype(float); SSTm = np.where(wsum > 0.5, (WO @ sstb) / np.maximum(wsum, 1e-9), np.nan)
    k6 = IL[600]; RH = np.minimum(100.0, 100 * Qm[k6] / qsat(Tm[k6], 600.0))
    PI = np.full(len(PL), np.nan); order = np.argsort(-LEV)
    for q in np.where(KEEP & np.isfinite(SSTm))[0]:
        lv = [l for l in order if 50 <= LEV[l] <= min(1000, MS[q])]
        TC = Tm[lv, q] - 273.15; R = Qm[lv, q] / (1 - Qm[lv, q]) * 1000.0
        try: PI[q] = float(bepi(SSTm[q] - 273.15, MS[q], LEV[lv], TC, R, CKCD=0.9, ascent_flag=0, diss_flag=1, V_reduc=0.8, ptop=50, miss_handle=1)[0])
        except Exception: pass
    tmp = fo + ".tmp.npz"
    np.savez(tmp, eta=ETA.astype(np.float32), rh=RH.astype(np.float32), pi=PI.astype(np.float32), vs=VS.astype(np.float32), sst=SSTm.astype(np.float32), keep=KEEP)
    os.replace(tmp, fo)
def days():
    """GENESIS_SCALE_PREREG.md: 320 fresh days (seed 161), 160 from 1980-99 and 160 from 2005-24, excluding all days in gpi_days..gpi7_days."""
    import glob as _g, os as _o
    used = {pd.Timestamp(_o.path.basename(f)[:10]) for d in ("gpi_days", "gpi2_days", "gpi3_days", "gpi4_days", "gpi5_days", "gpi6_days", "gpi7_days") for f in _g.glob(d + "/*.npz") if ".tmp" not in f}
    assert len(used) == 2240, len(used)
    rng = np.random.default_rng(161); out = []
    for y0, y1 in ((1980, 1999), (2005, 2024)):
        D = pd.DatetimeIndex([d for d in pd.date_range(f"{y0}-01-01", f"{y1}-12-31") if d not in used]); out += list(pd.DatetimeIndex(np.sort(rng.choice(D.values, 160, replace=False))))
    return out
if __name__ == "__main__":
    part, nparts = int(sys.argv[1]), int(sys.argv[2]); DD = days()[part::nparts]; t0 = time.time(); bad = []
    for n, d in enumerate(DD):
        try: one(d)
        except Exception as e: bad.append((str(d.date()), type(e).__name__, str(e)[:80]))
        if (n + 1) % 10 == 0: print(part, n + 1, len(DD), round(time.time() - t0), "s", flush=True)
    print("done", part, len(DD), "failed", len(bad), bad[:5], round(time.time() - t0), "s", flush=True)
