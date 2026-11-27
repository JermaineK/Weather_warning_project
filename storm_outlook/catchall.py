"""BETA, display only: high-sensitivity "catch-all" watch layer.

Points where the calibrated days 0-3 chance (same trees and calibration as the alerts, gate open, >5 deg from an active
storm) reaches a level set on 2017-19 to catch every storm (DESTRUCTIVE_CATCHALL_PREREG.md). On unseen years 2021-23 the
100% level flagged 269 of 271 storms >= 24 h ahead and its points saw a storm or >= 100 mm/day rain 1.4x as often as chance,
at about 15x the app's false alarms. It never changes alerts.
    import catchall; catchall.write(day, outdir, feat, storms)
Licence: PolyForm Noncommercial 1.0.0.
"""
import json, os
import numpy as np

LEVELS = {"all": 0.0007949939901958037, "p99": 0.0034083975695485117, "p95": 0.024088463979923286}   # catchall_frozen.json


def write(day, outdir, feat, storms):
    import outlook
    P, C = outlook.PTS, outlook.CFG
    doy = day.timetuple().tm_yday
    X = np.c_[feat.T.astype(np.float64), np.abs(P[:, 0]), np.full(len(P), np.cos(2 * np.pi * doy / 365.25)), np.full(len(P), np.sin(2 * np.pi * doy / 365.25))]
    z = C["platt"]["y03"][0] + C["platt"]["y03"][1] * outlook._logit(outlook._expit(outlook._trees_raw("y03", X)))
    s34 = [s for s in storms if s["kt"] >= C["storm_min_kt"]]
    dmin = (outlook.deg_dist(P[:, 0], P[:, 1], np.array([s["lat"] for s in s34]), np.array([s["lon"] for s in s34])).min(1) if s34 else np.full(len(P), 99.0))
    live = (feat[outlook.FEATURES.index("reach_3")] > C["gate_reach"]) & (dmin > 5.0)
    c = np.where(live, outlook._expit(z), 0.0)
    tier = np.where(c >= LEVELS["p95"], 3, np.where(c >= LEVELS["p99"], 2, np.where(c >= LEVELS["all"], 1, 0)))
    k = np.where(tier > 0)[0]
    out = dict(issued=f"{day:%Y-%m-%d}", note="BETA, display only. Catch-all watch (DESTRUCTIVE_CATCHALL_PREREG.md).",
               lat=P[k, 0].round(2).tolist(), lon=P[k, 1].round(2).tolist(), tier=tier[k].astype(int).tolist(),
               tiers={"1": "catch-all (100% level)", "2": "99% level", "3": "95% level"})
    tmp = os.path.join(outdir, f"catchall_{day:%Y%m%d}.json.tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    os.replace(tmp, os.path.join(outdir, f"catchall_{day:%Y%m%d}.json"))
    return out
