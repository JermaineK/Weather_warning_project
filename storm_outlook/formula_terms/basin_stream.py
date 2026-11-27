"""BASIN_FACTORS_PREREG.md: per 2.5-deg point and day, WAVE, UWEST, ZBIG, CONV from ERA5 850 hPa winds (00 UTC). No labels.
Days: all days in gpi_days..gpi5_days plus gpi6_stream.days(). Writes basin_days/<YYYY-MM-DD>.npz. Resumable.  python basin_stream.py <part> <nparts>"""
import os, sys, time, glob, numpy as np, pandas as pd, scipy.sparse as sp
_s = open("feed_stream.py", encoding="utf8").read(); exec(_s[:_s.find("def one(")])     # get, tidx, IL, lat, lon, a, OM, dl, dp, gc
OUT = "basin_days"; os.makedirs(OUT, exist_ok=True)
I0, I1 = int((90 - 40) / 0.25), int((90 + 40) / 0.25) + 1; BL = lat[I0:I1]; NB = (I1 - I0) * 1440
PLAT = np.r_[-np.arange(35, 2.4, -2.5), np.arange(2.5, 35.1, 2.5)]; PLON = np.arange(0, 360, 2.5)
PL = np.repeat(PLAT, len(PLON)); PO = np.tile(PLON, len(PLAT))
def disc(R):
    rows, cols, w = [], [], []; h = R / 111.2 + 0.3
    for q, (la0, lo0) in enumerate(zip(PL, PO)):
        i = np.where(np.abs(BL - la0) <= h)[0]; dlo = (lon - lo0 + 180) % 360 - 180; j = np.where(np.abs(dlo) <= h / np.cos(np.deg2rad(min(abs(la0) + h, 80))))[0]
        II, JJ = np.meshgrid(i, j, indexing="ij"); m = gc(la0, lo0, BL[II], lon[JJ]) <= R; II, JJ = II[m], JJ[m]
        rows += [q] * len(II); cols += list(II * 1440 + JJ); w += list(np.cos(np.deg2rad(BL[II])))
    W = sp.csr_matrix((w, (rows, cols)), shape=(len(PL), NB)); s = np.asarray(W.sum(1)).ravel(); return sp.diags(1 / s) @ W
W500, W1000 = disc(500.0), disc(1000.0)
def zsmooth(X, n):
    c = np.cumsum(np.concatenate([X[:, -n:], X, X[:, :n]], axis=1), axis=1); k = n // 2
    return (c[:, n + k:n + k + 1440] - c[:, n - (n - k):n - (n - k) + 1440]) / n
def one(day):
    fo = os.path.join(OUT, f"{day:%Y-%m-%d}.npz")
    if os.path.exists(fo): return
    ti = tidx(day); key = f"{ti}.0.0.0"; sh = (37, 721, 1440)
    U = get("u_component_of_wind", key, sh); V = get("v_component_of_wind", key, sh)
    u8, v8 = U[IL[850], I0 - 1:I1 + 1].astype(np.float64), V[IL[850], I0 - 1:I1 + 1].astype(np.float64); del U, V
    la = lat[I0 - 1:I1 + 1]; c = np.cos(np.deg2rad(la))[:, None]
    dvdx = (np.roll(v8, -1, 1) - np.roll(v8, 1, 1)) / (2 * dl); duc = (u8[:-2] * c[:-2] - u8[2:] * c[2:]) / (2 * (-dp))
    zeta = (dvdx[1:-1] - duc) / (a * c[1:-1]); sg = np.sign(BL)[:, None]
    dudx = (np.roll(u8, -1, 1) - np.roll(u8, 1, 1))[1:-1] / (2 * dl * a * c[1:-1])
    uu, vv = u8[1:-1], v8[1:-1]; vp = zsmooth(vv, 10) - zsmooth(vv, 80)
    WAVE = np.sqrt(W1000 @ (vp ** 2).ravel()); UWEST = W500 @ uu.ravel(); ZBIG = W1000 @ (sg * zeta).ravel(); CONV = W1000 @ (-dudx).ravel()
    tmp = fo + ".tmp.npz"; np.savez(tmp, WAVE=WAVE.astype(np.float32), UWEST=UWEST.astype(np.float32), ZBIG=ZBIG.astype(np.float32), CONV=CONV.astype(np.float32)); os.replace(tmp, fo)
def days():
    old = sorted({pd.Timestamp(os.path.basename(f)[:10]) for d in ("gpi_days", "gpi2_days", "gpi3_days", "gpi4_days", "gpi5_days") for f in glob.glob(d + "/*.npz") if ".tmp" not in f})
    _g = open("gpi6_stream.py", encoding="utf8").read(); ns = dict(np=np, pd=pd); exec(_g[_g.index("def days():"):_g.index('if __name__')], ns)
    return old + list(ns["days"]())
if __name__ == "__main__":
    part, nparts = int(sys.argv[1]), int(sys.argv[2]); DD = days()[part::nparts]; t0 = time.time(); bad = []
    for n, d in enumerate(DD):
        try: one(d)
        except Exception as e: bad.append((str(d.date()), type(e).__name__, str(e)[:80]))
        if (n + 1) % 20 == 0: print(part, n + 1, len(DD), round(time.time() - t0), "s", flush=True)
    print("done", part, len(DD), "failed", len(bad), bad[:5], round(time.time() - t0), "s", flush=True)
