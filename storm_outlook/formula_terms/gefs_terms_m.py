"""RESCALED_SPIN_PREREG.md: copy of gefs_terms.py with the day list read from rescaled_days.csv."""
"""GEFS_PORT_PREREG.md: GEFS control-run terms through the unchanged ERA5 term code (gpi_stream.py, basin_stream.py, trough_stream.py).
Only the data-access function is replaced: GEFS 0.5-deg fields are interpolated linearly to the ERA5 0.25-deg grid.
    python gefs_terms.py <lead_h> <part> <nparts> [rev]        -> gefs<lead>_gpi/, gefs<lead>_basin/, gefs<lead>_trough/<YYYY-MM-DD>.npz
    python gefs_terms.py days                                  -> gefs_port_days.csv (the 320 test days)"""
import os, sys, time, glob, numpy as np, pandas as pd, requests
MEM = os.environ.get("GEFS_MEM", "gec00"); MEMTAG = "" if MEM == "gec00" else "_" + MEM   # ENSEMBLE_PREREG.md
GB = "https://noaa-gefs-pds.s3.amazonaws.com"; ses = requests.Session()
ERA_DIRS = ("gpi_days", "gpi2_days", "gpi3_days", "gpi4_days", "gpi5_days", "gpi6_days", "gpi7_days", "gpi8_days", "gpi9_days", "gpi10_days", "gpi11_days", "gpi12_days")
def test_days():
    used = {pd.Timestamp(os.path.basename(f)[:10]) for d in ERA_DIRS for f in glob.glob(d + "/*.npz") if ".tmp" not in f}
    assert len(used) == 3840, len(used)
    D = pd.DatetimeIndex([d for d in pd.date_range("2021-01-01", "2025-10-31") if d not in used])
    return list(pd.DatetimeIndex(np.sort(np.random.default_rng(211).choice(D.values, 320, replace=False))))
def http(url, hdr=None):
    for k in range(6):
        try:
            r = ses.get(url, headers=hdr or {}, timeout=120); r.raise_for_status(); return r
        except Exception:
            if k == 5: raise
            time.sleep(5 * (k + 1))
def messages(stem, wanted):
    import eccodes
    idx = [l.split(":") for l in http(stem + ".idx").text.splitlines()]; out = {}
    for var, level in wanted:
        k = [i for i, a in enumerate(idx) if a[3] == var and a[4] == level][0]; a0 = int(idx[k][1])
        rng = f"bytes={a0}-{int(idx[k + 1][1]) - 1}" if k + 1 < len(idx) else f"bytes={a0}-"
        g = eccodes.codes_new_from_message(http(stem, {"Range": rng}).content)
        nj, ni = eccodes.codes_get(g, "Nj"), eccodes.codes_get(g, "Ni"); la0 = eccodes.codes_get(g, "latitudeOfFirstGridPointInDegrees")
        v = eccodes.codes_get_values(g).reshape(nj, ni).astype(np.float32); eccodes.codes_release(g)
        assert (nj, ni) == (361, 720) and abs(la0 - 90) < 1e-6, (nj, ni, la0)
        out[(var, level)] = v
    return out
def up(v):
    o = np.empty((721, 1440), np.float32); o[::2, ::2] = v; o[1::2, ::2] = 0.5 * (v[:-1] + v[1:])
    o[:, 1::2] = 0.5 * (o[:, ::2] + np.roll(o[:, ::2], -1, 1)); return o
GLEV = [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 70, 50]
A_LEV = {1000, 925, 850, 700, 500, 250, 200, 100, 50}
CUR = {}
def load(day, lead, ELEV):
    init = pd.Timestamp(day) - pd.Timedelta(hours=lead)
    stem = lambda p: f"{GB}/gefs.{init:%Y%m%d}/00/atmos/{p}/{MEM}.t00z.{p[:6]}.0p50.f{lead:03d}"
    wa = [("UGRD", "850 mb"), ("VGRD", "850 mb"), ("UGRD", "200 mb"), ("VGRD", "200 mb"), ("PRMSL", "mean sea level")] + [("TMP", f"{l} mb") for l in GLEV if l in A_LEV]
    wb = [("TMP", "surface"), ("LAND", "surface"), ("ICEC", "surface")] + [("TMP", f"{l} mb") for l in GLEV if l not in A_LEV] + [("SPFH", f"{l} mb") for l in GLEV]
    M = messages(stem("pgrb2ap5"), wa); M.update(messages(stem("pgrb2bp5"), wb))
    n = len(ELEV); U = np.full((n, 721, 1440), np.nan, np.float32); V = U.copy(); T = U.copy(); Q = U.copy()
    for p in (850, 200):
        i = ELEV.index(p); U[i] = up(M[("UGRD", f"{p} mb")]); V[i] = up(M[("VGRD", f"{p} mb")])
    have = sorted(GLEV)
    for i, p in enumerate(ELEV):
        if p < 50 or p > 1000: continue
        if p in GLEV: T[i] = up(M[("TMP", f"{p} mb")]); Q[i] = up(M[("SPFH", f"{p} mb")]); continue
        lo = max(l for l in have if l < p); hi = min(l for l in have if l > p); w = (np.log(p) - np.log(lo)) / (np.log(hi) - np.log(lo))
        for X, var in ((T, "TMP"), (Q, "SPFH")):
            X[i] = up((1 - w) * M[(var, f"{lo} mb")] + w * M[(var, f"{hi} mb")])
    sst = M[("TMP", "surface")].copy(); sst[(M[("LAND", "surface")] >= 0.5) | (M[("ICEC", "surface")] >= 0.15)] = np.nan
    CUR.clear(); CUR.update(U=U, V=V, T=T, Q=Q, SST=up(sst), MSL=up(M[("PRMSL", "mean sea level")]))
MAP = {"u_component_of_wind": "U", "v_component_of_wind": "V", "temperature": "T", "specific_humidity": "Q", "sea_surface_temperature": "SST", "mean_sea_level_pressure": "MSL"}
def gget(v, key, shape):
    X = CUR[MAP[v]]; assert X.shape == tuple(shape), (v, X.shape, shape); return X
def term_ns(script, out):
    src = open(script, encoding="utf8").read(); ns = {"__name__": "gefs_port", "__file__": os.path.abspath(script)}
    exec(src[:src.index("\ndef days():")], ns); ns["get"] = gget; ns["tidx"] = lambda d: 0; ns["OUT"] = out; os.makedirs(out, exist_ok=True); return ns
if __name__ == "__main__":
    if sys.argv[1] == "days":
        pd.DataFrame(dict(day=[d.date() for d in test_days()])).to_csv("gefs_port_days.csv", index=False); sys.exit()
    lead, part, nparts = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    DD = [pd.Timestamp(d) for d in pd.read_csv("rescaled_days.csv").day][part::nparts]
    if len(sys.argv) > 4 and sys.argv[4] == "rev": DD = DD[::-1]
    NS = [term_ns(s, f"gefs{lead:03d}{MEMTAG}_{k}") for s, k in (("gpi_stream.py", "gpi"), ("basin_stream.py", "basin"), ("trough_stream.py", "trough"))]
    ELEV = [int(x) for x in NS[0]["LEV"]]; t0 = time.time(); bad = []
    for n, d in enumerate(DD):
        if all(os.path.exists(os.path.join(ns["OUT"], f"{d:%Y-%m-%d}.npz")) for ns in NS): continue
        try:
            load(d, lead, ELEV)
            for ns in NS: ns["one"](d)
        except Exception as e: bad.append((str(d.date()), type(e).__name__, str(e)[:100]))
        if (n + 1) % 10 == 0: print(lead, part, n + 1, len(DD), round(time.time() - t0), "s", flush=True)
    print("done", lead, part, len(DD), "failed", len(bad), bad[:5], round(time.time() - t0), "s", flush=True)
