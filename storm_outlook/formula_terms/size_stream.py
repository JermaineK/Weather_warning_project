"""GENESIS_SCALE_PREREG.md: per 2.5-deg point and day, ZPK1 and ZPK3 = max within 556 km of s*zeta850 smoothed (Gaussian sigma 0.5 / 1.5 deg, grid degrees, wrap in lon).
Days: gpi_days..gpi7_days plus gpi8_stream.days(). Writes size_days/<YYYY-MM-DD>.npz. Resumable.  python size_stream.py <part> <nparts>"""
import os, sys, time, glob, numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter
_s = open("feed_stream.py", encoding="utf8").read(); exec(_s[:_s.find("def one(")])     # get, tidx, IL, lat, lon, a, OM, dl, dp, gc
OUT = "size_days"; os.makedirs(OUT, exist_ok=True)
I0, I1 = int((90 - 40) / 0.25), int((90 + 40) / 0.25) + 1; BL = lat[I0:I1]
PLAT = np.r_[-np.arange(35, 2.4, -2.5), np.arange(2.5, 35.1, 2.5)]; PLON = np.arange(0, 360, 2.5); PL = np.repeat(PLAT, len(PLON)); PO = np.tile(PLON, len(PLAT))
def build():
    idx, starts = [], []; n = 0
    for la0, lo0 in zip(PL, PO):
        i = np.where(np.abs(BL - la0) <= 5.3)[0]; dlo = (lon - lo0 + 180) % 360 - 180; j = np.where(np.abs(dlo) <= 5.3 / np.cos(np.deg2rad(min(abs(la0) + 5.3, 80))))[0]
        II, JJ = np.meshgrid(i, j, indexing="ij"); m = gc(la0, lo0, BL[II], lon[JJ]) <= 556.0; f = (II[m] * 1440 + JJ[m]); starts.append(n); idx.append(f); n += len(f)
    return np.concatenate(idx), np.array(starts)
IDX, ST = build()
def one(day):
    fo = os.path.join(OUT, f"{day:%Y-%m-%d}.npz")
    if os.path.exists(fo): return
    ti = tidx(day); key = f"{ti}.0.0.0"; sh = (37, 721, 1440)
    U = get("u_component_of_wind", key, sh); V = get("v_component_of_wind", key, sh)
    u8, v8 = U[IL[850], I0 - 1:I1 + 1].astype(np.float64), V[IL[850], I0 - 1:I1 + 1].astype(np.float64); del U, V
    la = lat[I0 - 1:I1 + 1]; c = np.cos(np.deg2rad(la))[:, None]
    dvdx = (np.roll(v8, -1, 1) - np.roll(v8, 1, 1)) / (2 * dl); duc = (u8[:-2] * c[:-2] - u8[2:] * c[2:]) / (2 * (-dp))
    sz = np.sign(BL)[:, None] * (dvdx[1:-1] - duc) / (a * c[1:-1])
    out = {}
    for nm, sg in (("ZPK1", 2.0), ("ZPK3", 6.0)):
        f = gaussian_filter(sz, sigma=sg, mode=("nearest", "wrap")).ravel(); out[nm] = np.maximum.reduceat(f[IDX], ST).astype(np.float32)
    tmp = fo + ".tmp.npz"; np.savez(tmp, **out); os.replace(tmp, fo)
def days():
    old = sorted({pd.Timestamp(os.path.basename(f)[:10]) for d in ("gpi_days", "gpi2_days", "gpi3_days", "gpi4_days", "gpi5_days", "gpi6_days", "gpi7_days") for f in glob.glob(d + "/*.npz") if ".tmp" not in f})
    _g = open("gpi8_stream.py", encoding="utf8").read(); ns = dict(np=np, pd=pd); exec(_g[_g.index("def days():"):_g.index('if __name__')], ns)
    return old + list(ns["days"]())
if __name__ == "__main__":
    part, nparts = int(sys.argv[1]), int(sys.argv[2]); DD = days()[part::nparts]; t0 = time.time(); bad = []
    for n, d in enumerate(DD):
        try: one(d)
        except Exception as e: bad.append((str(d.date()), type(e).__name__, str(e)[:80]))
        if (n + 1) % 20 == 0: print(part, n + 1, len(DD), round(time.time() - t0), "s", flush=True)
    print("done", part, len(DD), "failed", len(bad), bad[:5], round(time.time() - t0), "s", flush=True)
