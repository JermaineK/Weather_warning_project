"""BETA, display only: formula outlook layer (formula 1.0.0).

The storm-formation formula tested in LEAD_TIME_RESULT.md and ENSEMBLE_RESULT.md, run on today's GEFS:
  - now, +1 day, +2 days: full formula (moisture, spin, shear, warm water, humidity, wave/trough factors, two shape steps,
    small-scale spin in the blind spot), GEFS control run;
  - +3 days: steps-only formula (no small-scale spin), averaged over 5 GEFS members (control + 4 perturbed).
Each value is the chance that a storm (>= 34 kt) forms within 556 km of the point in the 72 h after the map's valid time.
Points within 556 km of an active storm at issue time are left out, as in the tests.

Tested on 966 unseen days (2021-25): AUC 0.956 now, 0.954 / 0.952 at +1 / +2 days, 0.952 at +3 days with the ensemble.
Absolute chances carry about +-20% year-to-year uncertainty (ENSEMBLE_RECAL_RESULT.md), so the page shows risk bands.
It never changes alerts.
    import formula; formula.run(day, outdir, storms)
Term code: formula_terms/ (byte-identical copies of the research scripts). Constants: model/formula_frozen.json.
Licence: PolyForm Noncommercial 1.0.0.
"""
import contextlib, glob, json, os, shutil
import numpy as np
import pandas as pd
from scipy.special import expit

HERE = os.path.dirname(os.path.abspath(__file__)); TD = os.path.join(HERE, "formula_terms")
FZ = json.load(open(os.path.join(HERE, "model", "formula_frozen.json")))
PLAT = np.r_[-np.arange(35, 2.4, -2.5), np.arange(2.5, 35.1, 2.5)]; PLON = np.arange(0, 360, 2.5); NLON = 144
PL = np.repeat(PLAT, NLON); PO = np.tile(PLON, len(PLAT))
MEMBERS = ("gec00", "gep01", "gep02", "gep03", "gep04")
LEADS = {0: (("gec00",), "FULL"), 24: (("gec00",), "FULL"), 48: (("gec00",), "FULL"), 72: (MEMBERS, "F12")}
BANDS = ((0.20, 4, "very high"), (0.05, 3, "high"), (0.01, 2, "raised"))   # chance of a storm within 556 km in 72 h; below 1% = low


# ---------------------------------------------------------------- terms (research code, unchanged)
@contextlib.contextmanager
def _in(d):
    old = os.getcwd(); os.chdir(d)
    try: yield
    finally: os.chdir(old)

_ENV = {}
def _env(scratch):
    """Load gefs_terms_m.py definitions and the four term namespaces once per process."""
    if _ENV: return _ENV
    with _in(TD):
        src = open("gefs_terms_m.py", encoding="utf8").read(); g = {"__name__": "formula_terms", "__file__": os.path.join(TD, "gefs_terms_m.py")}
        exec(src[:src.index('\nif __name__ == "__main__":')], g)
        NS = {k: g["term_ns"](s, scratch) for s, k in (("gpi_stream.py", "gpi"), ("basin_stream.py", "basin"), ("trough_stream.py", "trough"))}
        s2 = open("size_stream.py", encoding="utf8").read(); ns = {"__name__": "port_size", "__file__": os.path.join(TD, "size_stream.py")}
        exec(s2[:s2.index("\ndef days():")], ns)
    CURW = {}; ns["IL"] = {850: 0}; ns["tidx"] = lambda d: 0
    ns["get"] = lambda v, key, shape: CURW["U" if v == "u_component_of_wind" else "V"]
    NS["size"] = ns; _ENV.update(g=g, NS=NS, CURW=CURW, ELEV=[int(x) for x in NS["gpi"]["LEV"]]); return _ENV

def compute_terms(valid, lead, mem, cache):
    """Terms for one valid day, lead (h) and member -> {kind: npz path}. Same calls as gefs_terms_m.py / port_size_terms_m.py."""
    valid = pd.Timestamp(valid); cache = os.path.abspath(cache); dirs = {k: os.path.join(cache, f"f{lead:03d}_{mem}", k) for k in ("gpi", "basin", "trough", "size")}
    paths = {k: os.path.join(d, f"{valid:%Y-%m-%d}.npz") for k, d in dirs.items()}
    if all(os.path.exists(p) for p in paths.values()): return paths
    for d in dirs.values(): os.makedirs(d, exist_ok=True)
    E = _env(dirs["gpi"]); g = E["g"]
    with _in(TD):
        g["MEM"] = mem; g["load"](valid, lead, E["ELEV"])
        for k in ("gpi", "basin", "trough"): E["NS"][k]["OUT"] = dirs[k]; E["NS"][k]["one"](valid)
        init = valid - pd.Timedelta(hours=lead)
        M = g["messages"](f"{g['GB']}/gefs.{init:%Y%m%d}/00/atmos/pgrb2ap5/{mem}.t00z.pgrb2a.0p50.f{lead:03d}", [("UGRD", "850 mb"), ("VGRD", "850 mb")])
        E["CURW"]["U"] = g["up"](M[("UGRD", "850 mb")])[None]; E["CURW"]["V"] = g["up"](M[("VGRD", "850 mb")])[None]
        E["NS"]["size"]["OUT"] = dirs["size"]; E["NS"]["size"]["one"](valid)
    return paths


# ---------------------------------------------------------------- formula (frozen)
def gc(la1, lo1, la2, lo2):
    p1, p2 = np.deg2rad(la1), np.deg2rad(la2)
    return 6371.0 * np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(np.deg2rad(lo2 - lo1)), -1, 1))

def basin(la, lo):
    return np.select([(la > 0) & (lo >= 260) & (lo < 360), (la > 0) & (lo >= 180) & (lo < 260), (la > 0) & (lo >= 100) & (lo < 180), (la > 0) & (lo >= 40) & (lo < 100),
                      (la < 0) & (lo >= 20) & (lo < 135), (la < 0) & (lo >= 135) & (lo < 240)], ["N Atlantic", "E Pacific", "W Pacific", "N Indian", "S Indian", "S Pacific"], "other")

def ns_basis(x, knots):
    k = np.asarray(knots, float); K = len(k); d = lambda j: (np.maximum(0, x - k[j]) ** 3 - np.maximum(0, x - k[-1]) ** 3) / (k[-1] - k[j])
    return np.c_[tuple([x] + [d(j) - d(K - 2) for j in range(K - 2)])]

def cl(X, b): return 1 - np.exp(-np.exp(np.clip(X @ np.asarray(b), -30, 5)))

def rows(paths, valid, storms):
    """Per-point table for one valid day (the research rows + with_dirs + small-spin steps, without labels)."""
    valid = pd.Timestamp(valid); z = {k: np.load(p) for k, p in paths.items()}; g = z["gpi"]
    s34 = [s for s in (storms or []) if s.get("kt", 0) >= FZ["storm_min_kt"]]
    near = (gc(PL[:, None], PO[:, None], np.array([s["lat"] for s in s34])[None], (np.array([s["lon"] for s in s34]) % 360)[None]) <= FZ["exclude_km"]).any(1) if s34 else np.zeros(len(PL), bool)
    D = pd.DataFrame(dict(lat=PL, lon=PO, eta=g["eta"].astype(float), rh=g["rh"].astype(float), pi=np.maximum(g["pi"].astype(float), 10), vs=g["vs"].astype(float), sst=g["sst"].astype(float)))
    for k in FZ["FAC"]: D[k] = z["basin"][k].astype(float)
    D["lnWAVE"] = np.log(np.maximum(D.WAVE, 0.05)); dt = z["trough"]["DT"].astype(float)
    D["lnDT"] = np.log1p(dt); D["TSTR"] = z["trough"]["TSTR"].astype(float); D["MW"] = z["trough"]["MW"].astype(float)
    z1, z3 = z["size"]["ZPK1"].astype(float), z["size"]["ZPK3"].astype(float); m = FZ["spin_map"]
    D["PEAK"] = np.interp(z1 * 1e5, m["PEAK_src"], m["PEAK_dst"]); D["CONC"] = np.interp((z1 - z3) * 1e5, m["CONC_src"], m["CONC_dst"])
    D.loc[~np.isfinite(z1), "PEAK"] = np.nan; D.loc[~(np.isfinite(z1) & np.isfinite(z3)), "CONC"] = np.nan
    keep = g["keep"].astype(bool) & ~near & np.isfinite(g["pi"]) & D[FZ["FAC"] + FZ["SC"] + ["PEAK", "CONC"]].notna().all(1).values
    D["near_storm"] = near; D = D[keep].reset_index(drop=True)
    D["x1"] = np.log(1e5 * D.eta.clip(lower=1e-7)); D["x2"] = np.log(D.rh.clip(lower=1) / 50); D["x3"] = np.log(D.pi / 70); D["x4"] = np.log1p(0.1 * D.vs)
    hs = np.where(D.lat > 0, valid.month in (7, 8, 9, 10), valid.month in (1, 2, 3, 4))
    D["cell"] = pd.Series(basin(D.lat.values, D.lon.values)) + "|" + np.where(hs, "peak", "off-peak"); return D

def design_X(D):
    """Columns: [base block (k0)] + weak-wave step, off-season/subtropical step, PEAK, CONC, S, PEAK*S, CONC*S."""
    sp = FZ["spec"]; s = D.sst.values - 273.15
    des = [np.ones(len(D)), D.x1.values, D.x3.values, D.x4.values, np.maximum(0, D.x4.values - sp["k4"]), np.log(np.clip(expit((s - sp["T0"]) / sp["w"]), 1e-12, 1))] + list(ns_basis(D.x2.values, sp["knots"]).T)
    def blk(P):
        mu, sd = P["mu"], P["sd"]
        return des + [((D[c] - mu[c]) / sd[c]).values for c in FZ["zc"] + FZ["SC"]] + [D.MW.values] + [(D.cell.values == c).astype(float) for c in P["cells"]]
    B, M = FZ["blind"], FZ["main"]
    S = (cl(np.c_[tuple(blk(B))], B["coef"]) < B["threshold"]).astype(float)
    rw = (D.lnWAVE.values < M["q50"]).astype(float); f2 = (D.cell.str.endswith("off-peak").values | (D.lat.abs().values >= 20)).astype(float)
    p = (D.PEAK.values - M["peak"][0]) / M["peak"][1]; c = (D.CONC.values - M["conc"][0]) / M["conc"][1]
    return np.c_[tuple(blk(M) + [rw, f2, p, c, S, p * S, c * S])], S.astype(bool)

def probs(D):
    X, S = design_X(D); k0 = FZ["main"]["k0"]
    return dict(FULL=cl(X[:, :k0 + 7], FZ["main"]["coef_FULL"]), F12=cl(X[:, :k0 + 2], FZ["main"]["coef_F12"]), S=S)

def band(p):
    for t, i, _ in BANDS:
        if p >= t: return i
    return 1


# ---------------------------------------------------------------- daily output
def run(day, outdir, storms, log=print, keep_cache_days=2):
    """Write formula_<YYYYMMDD>.json for the 00 UTC cycle of `day` (datetime/date). Each lead is computed independently."""
    day = pd.Timestamp(f"{pd.Timestamp(day):%Y-%m-%d}"); root = os.path.join(outdir, "formula_cache"); cache = os.path.join(root, f"{day:%Y%m%d}")
    out = dict(issued=f"{day:%Y-%m-%d}", version=FZ["version"], note="BETA, display only. Formula outlook (LEAD_TIME_RESULT.md, ENSEMBLE_RESULT.md).",
               bands={str(i): n for _, i, n in BANDS} | {"1": "low"}, band_limits=[t for t, _, _ in BANDS][::-1], leads={})
    for L, (mems, model) in LEADS.items():
        valid = day + pd.Timedelta(hours=L)
        try:
            P, F12, Dref = [], [], None
            for m in mems:
                D = rows(compute_terms(valid, L, m, cache), valid, storms); r = probs(D)
                key = D.lat.round(2).astype(str) + "|" + D.lon.round(2).astype(str)
                if Dref is None: Dref, kref = D, key
                s = pd.Series(r[model], index=key.values).reindex(kref.values); P.append(s.values); F12.append(pd.Series(r["F12"], index=key.values).reindex(kref.values).values)
            p = np.nanmean(np.vstack(P), 0) if len(P) > 1 else P[0]; nmem = np.isfinite(np.vstack(P)).sum(0)
            ok = np.isfinite(p) & (nmem == len(mems)); D = Dref[ok]; p = p[ok]; f12 = np.nanmean(np.vstack(F12), 0)[ok]
            spin = (model == "FULL") & (p >= 0.01) & (p >= 2 * f12)          # small-scale spin at least doubled the chance (descriptive)
            sel = np.where(p >= 0.001)[0]
            out["leads"][str(L // 24)] = dict(valid=f"{valid:%Y-%m-%d}", model="full formula, GEFS control" if model == "FULL" else f"steps-only formula, mean of {len(mems)} GEFS members",
                                              n_points=int(len(p)), lat=D.lat.values[sel].round(2).tolist(), lon=D.lon.values[sel].round(2).tolist(),
                                              p=np.round(p[sel], 4).tolist(), band=[band(x) for x in p[sel]], spin=spin[sel].astype(int).tolist())
            log(f"{day:%Y-%m-%d}: formula +{L // 24} d done ({len(sel)} points >= 0.1%)")
        except Exception as e:
            out["leads"][str(L // 24)] = dict(valid=f"{valid:%Y-%m-%d}", error=f"{type(e).__name__}: {str(e)[:120]}")
            log(f"{day:%Y-%m-%d}: formula +{L // 24} d skipped ({type(e).__name__}: {str(e)[:80]})")
    tmp = os.path.join(outdir, f"formula_{day:%Y%m%d}.json.tmp")
    with open(tmp, "w") as f: json.dump(out, f, separators=(",", ":"))
    os.replace(tmp, os.path.join(outdir, f"formula_{day:%Y%m%d}.json"))
    for d in sorted(glob.glob(os.path.join(root, "*")))[:-keep_cache_days]: shutil.rmtree(d, ignore_errors=True)
    return out
