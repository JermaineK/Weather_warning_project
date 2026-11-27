"""TROUGH_POSITION_PREREG.md: per 2.5-deg point and day, distance DT (deg) to own-hemisphere 850 hPa trough axis (max of s*zeta, 10-deg lon x 2-deg lat running mean, 3-25 deg),
axis strength TSTR (1e-5/s) and monsoon-westerly flag MW (max mean u between equator and axis > 2 m/s). Days: gpi_days..gpi9_days + gpi10_stream.days(). Writes trough_days/<date>.npz. Resumable.
    python trough_stream.py <part> <nparts> [rev]"""
import os, sys, time, glob, numpy as np, pandas as pd
from scipy.ndimage import uniform_filter1d
_s = open("feed_stream.py", encoding="utf8").read(); exec(_s[:_s.find("def one(")])     # get, tidx, IL, lat, lon, a, OM, dl, dp, gc
OUT = "trough_days"; os.makedirs(OUT, exist_ok=True)
I0, I1 = int((90 - 40) / 0.25), int((90 + 40) / 0.25) + 1; BL = lat[I0:I1]
PLAT = np.r_[-np.arange(35, 2.4, -2.5), np.arange(2.5, 35.1, 2.5)]; PLON = np.arange(0, 360, 2.5); PL = np.repeat(PLAT, len(PLON)); PO = np.tile(PLON, len(PLAT))
JJ = np.round(PO / 0.25).astype(int) % 1440; NHr = np.where((BL >= 3) & (BL <= 25))[0]; SHr = np.where((BL <= -3) & (BL >= -25))[0]
def axis(zs, ub, rows, north):
    k = np.argmax(zs[rows], axis=0); phi = BL[rows][k]; tstr = zs[rows][k, np.arange(1440)]
    eq = np.where(BL >= 0)[0] if north else np.where(BL <= 0)[0]; L = BL[eq][:, None]
    m = (L <= phi[None]) if north else (L >= phi[None]); mw = np.where(m, ub[eq], -99).max(0) > 2.0
    return phi, tstr, mw
def one(day):
    fo = os.path.join(OUT, f"{day:%Y-%m-%d}.npz")
    if os.path.exists(fo): return
    ti = tidx(day); key = f"{ti}.0.0.0"; sh = (37, 721, 1440)
    U = get("u_component_of_wind", key, sh); V = get("v_component_of_wind", key, sh)
    u8, v8 = U[IL[850], I0 - 1:I1 + 1].astype(np.float64), V[IL[850], I0 - 1:I1 + 1].astype(np.float64); del U, V
    la = lat[I0 - 1:I1 + 1]; c = np.cos(np.deg2rad(la))[:, None]
    dvdx = (np.roll(v8, -1, 1) - np.roll(v8, 1, 1)) / (2 * dl); duc = (u8[:-2] * c[:-2] - u8[2:] * c[2:]) / (2 * (-dp))
    sz = np.sign(BL)[:, None] * (dvdx[1:-1] - duc) / (a * c[1:-1])
    sm = lambda X: uniform_filter1d(uniform_filter1d(X, 41, axis=1, mode="wrap"), 9, axis=0, mode="nearest")
    zs = sm(sz); ub = sm(u8[1:-1])
    pN, tN, mN = axis(zs, ub, NHr, True); pS, tS, mS = axis(zs, ub, SHr, False)
    nh = PL > 0; phi = np.where(nh, pN[JJ], pS[JJ]); DT = np.abs(PL - phi); TSTR = np.where(nh, tN[JJ], tS[JJ]) * 1e5; MW = np.where(nh, mN[JJ], mS[JJ])
    tmp = fo + ".tmp.npz"; np.savez(tmp, DT=DT.astype(np.float32), TSTR=TSTR.astype(np.float32), MW=MW, phiN=pN.astype(np.float32), phiS=pS.astype(np.float32)); os.replace(tmp, fo)
def days():
    old = sorted({pd.Timestamp(os.path.basename(f)[:10]) for d in ("gpi_days", "gpi2_days", "gpi3_days", "gpi4_days", "gpi5_days", "gpi6_days", "gpi7_days", "gpi8_days", "gpi9_days") for f in glob.glob(d + "/*.npz") if ".tmp" not in f})
    _g = open("gpi10_stream.py", encoding="utf8").read(); ns = dict(np=np, pd=pd); exec(_g[_g.index("def days():"):_g.index('if __name__')], ns)
    return old + list(ns["days"]())
if __name__ == "__main__":
    part, nparts = int(sys.argv[1]), int(sys.argv[2]); DD = days()[part::nparts]
    if len(sys.argv) > 3 and sys.argv[3] == "rev": DD = DD[::-1]
    t0 = time.time(); bad = []
    for n, d in enumerate(DD):
        try: one(d)
        except Exception as e: bad.append((str(d.date()), type(e).__name__, str(e)[:80]))
        if (n + 1) % 20 == 0: print(part, n + 1, len(DD), round(time.time() - t0), "s", flush=True)
    print("done", part, len(DD), "failed", len(bad), bad[:5], round(time.time() - t0), "s", flush=True)
