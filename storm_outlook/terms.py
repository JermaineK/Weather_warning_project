"""Physical term breakdown for the dashboard ("why is this region flagged?").

The alerts themselves come from the tree models. This module evaluates the simpler, pre-registered physical formulas
(TERMS_ROUND3_RESULT.md) at every point, and reports each term's contribution in log-odds relative to a typical point.
That lets a user see which ingredients push a place up or down. Formula chances are shown alongside the trees' chances;
the trees remain the forecast.
Licence: PolyForm Noncommercial 1.0.0.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, "config.json")))["terms"]


def _sig(x):
    return 1 / (1 + np.exp(-x))


def tropical(feat, pts, sst5=None):
    """feat: 30 x n app features (lead-3 rows 0-5: z5, zmax, sh, reach, pw, q7); sst5: 5-deg mean surface temperature (K) or None."""
    c, ref, g = T["tropical"]["coef"], T["tropical"]["reference"], T["tropical"]["gate"]
    z5, sh, reach, q7 = feat[0].astype(float), feat[2].astype(float), feat[3].astype(float), feat[5].astype(float)
    fabs = 2 * 7.292e-5 * np.abs(np.sin(np.deg2rad(pts[:, 0])))
    N = z5 * 1e5
    v = dict(le=np.log(np.maximum(z5 + fabs, 1e-6) * 1e5), alat=np.abs(pts[:, 0]) / 10, lH=np.log(np.maximum(q7 * 100, 1e-3)),
             S=np.log1p(sh / 10), lR=np.log(np.clip(reach, 1e-6, 1)))
    lock = _sig((N - g["N_star"]) / g["width"])                     # 0 = shear fully felt, 1 = shear switched off
    have_sst = sst5 is not None
    v["E"] = np.log1p(np.maximum(sst5 - 299.65, 0)) if have_sst else np.full(len(pts), ref["E"])
    ref_lock = _sig((ref["N"] - g["N_star"]) / g["width"])
    contrib = dict(
        spin=c["spin"] * (v["le"] - ref["le"]),
        latitude=c["lat"] * (v["alat"] - ref["alat"]),
        humidity=c["humidity"] * (v["lH"] - ref["lH"]),
        shear=c["shear"] * (v["S"] * (1 - lock) - ref["S"] * (1 - ref_lock)),
        instability=c["reach"] * (v["lR"] - ref["lR"]),
        ocean_heat=c["sst"] * (v["E"] - ref["E"]))
    z = c["intercept"] + c["spin"] * v["le"] + c["lat"] * v["alat"] + c["humidity"] * v["lH"] + c["shear"] * v["S"] * (1 - lock) + c["reach"] * v["lR"] + c["sst"] * v["E"]
    raw = dict(spin_1e5=N, shear_ms=sh, humidity_gkg=q7 * 1000, instability_share=reach, sst_c=(sst5 - 273.15) if have_sst else np.full(len(pts), np.nan))
    return contrib, _sig(z), lock, raw, have_sst


def bomb(feat17, bp):
    c, ref = T["bomb"]["coef"], T["bomb"]["reference"]
    X = feat17.astype(float); fb = 14.584 * np.abs(np.sin(np.deg2rad(bp[:, 0])))
    up = lambda v, k: np.roll(v.reshape(-1, 144), k, axis=1).ravel()
    le = np.log(np.maximum(X[7] + fb, 0.1 * fb)); lB = np.log(np.maximum(X[10], 0.05)); lPW = np.log(np.maximum(X[16], 0.5)); lJ = np.log(np.maximum(X[13], 5))
    u = c["up"]; tr = u[0] * up(X[7], 2) / 10 + u[1] * up(X[7], 4) / 10 + u[2] * up(X[8], 2) / 10 + u[3] * up(X[8], 4) / 10
    low = c["low"] * X[0] / 10 + c["low_x_trough"] * up(X[7], 2) / 10 * np.maximum(-X[0], 0) / 10
    contrib = dict(spin=c["spin"] * (le - ref["le"]), temperature_contrast=c["baroclinicity"] * (lB - ref["lB"]), moisture=c["moisture"] * (lPW - ref["lPW"]),
                   jet=c["jet"] * (lJ - ref["lJ"]), upstream_trough=tr - ref["up"], surface_low=low - ref["low"])
    z = c["intercept"] - np.log(20.0) + c["spin"] * le + c["baroclinicity"] * lB + c["moisture"] * lPW + c["jet"] * lJ + tr + low
    return contrib, _sig(z)


def write(day, outdir, feat, pts, p03, live, sst5, write_json, alert_mask=None):
    contrib, pf, lock, raw, have_sst = tropical(feat, pts, sst5)
    keep = np.where(live)[0]
    rnd = lambda a, k=2: [round(float(x), k) if np.isfinite(x) else None for x in a]
    out = dict(issued=f"{day:%Y-%m-%d} 00 UTC", sst_available=bool(have_sst), lat=rnd(pts[keep, 0], 1), lon=rnd(pts[keep, 1], 1),
               tree_chance=rnd(p03[keep], 4), formula_chance=rnd(pf[keep], 4), lock_in=rnd(lock[keep]),
               contrib={k: rnd(v[keep]) for k, v in contrib.items()}, raw={k: rnd(v[keep], 1 if k != "instability_share" else 2) for k, v in raw.items()})
    write_json(os.path.join(outdir, f"terms_{day:%Y%m%d}.json"), out, separators=(",", ":"))
