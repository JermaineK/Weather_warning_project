"""Dashboard extras (display only; never changes alerts or verification):

1. Forecast-hour view: the physical formula and its ingredients at each forecast lead (3, 24, 48, 96, 120 h), from the
   control-run features the outlook already archives (archive/features_YYYYMMDD.npz). No extra download.
2. Ensemble agreement: for the latest day only, the share of GEFS ensemble members whose 5-degree net spin at 850 hPa
   exceeds the lock-in threshold (1.17e-5 s^-1, BATHTUB_GATE_RESULT.md), at 24-120 h. About 2.7 MB per member per day.

    python forecast_view.py --out <folder> [--members 10]
Licence: PolyForm Noncommercial 1.0.0.
"""
import argparse, json, os, sys
from datetime import datetime, timezone
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import outlook as OL, terms as TM

FLEADS = [3, 24, 48, 96, 120]          # order of the archived feature blocks (6 features each)
ELEADS = [24, 48, 72, 96, 120]
LOCK = TM.T["tropical"]["gate"]["N_star"]


def _w(path, obj):
    OL._write_json(path, obj, separators=(",", ":"))


def leads_view(outdir, day):
    fa = os.path.join(outdir, "archive", f"features_{day:%Y%m%d}.npz")
    tj = os.path.join(outdir, f"terms_{day:%Y%m%d}.json")
    if not os.path.exists(fa) or not os.path.exists(tj):
        return False
    X = np.load(fa)["X"].astype(np.float64)
    T = json.load(open(tj))
    sst = np.array([np.nan if v is None else v for v in T["raw"]["sst_c"]], float) + 273.15
    pts = OL.PTS
    idx = {(round(a, 1), round(b, 1)): i for i, (a, b) in enumerate(pts)}
    keep = np.array([idx[(round(a, 1), round(b, 1))] for a, b in zip(T["lat"], T["lon"])], int)
    out = dict(issued=T["issued"], leads=FLEADS, lat=T["lat"], lon=T["lon"], chance=[], lock_in=[], spin=[], shear=[])
    for k, h in enumerate(FLEADS):
        f6 = np.zeros((30, len(pts)), np.float64); f6[0:6] = X[6 * k:6 * k + 6]
        s5 = np.full(len(pts), np.nan); s5[keep] = sst
        _, pf, lock, raw, _ = TM.tropical(f6, pts, None if np.isnan(s5[keep]).all() else np.where(np.isnan(s5), 299.65, s5))
        out["chance"].append([round(float(x), 4) for x in pf[keep]]); out["lock_in"].append([round(float(x), 2) for x in lock[keep]])
        out["spin"].append([round(float(x), 2) for x in raw["spin_1e5"][keep]]); out["shear"].append([round(float(x), 1) for x in raw["shear_ms"][keep]])
    _w(os.path.join(outdir, f"leads_{day:%Y%m%d}.json"), out)
    return True


def ensemble_view(outdir, day, members=10):
    base = OL.CFG["sources"]["gefs"]; pts = OL.PTS
    share = np.zeros((len(ELEADS), len(pts))); mean = np.zeros_like(share); n_ok = 0
    for m in range(1, members + 1):
        try:
            Z = []
            for h in ELEADS:
                stem = f"{base}/gefs.{day:%Y%m%d}/00/atmos/pgrb2ap5/gep{m:02d}.t00z.pgrb2a.0p50.f{h:03d}"
                u, v = OL.grib_field(stem, "UGRD", "850 mb", h), OL.grib_field(stem, "VGRD", "850 mb", h)
                z = ((np.roll(v, -1, 1) - np.roll(v, 1, 1)) / (2 * OL._DD) - np.gradient(u * np.cos(OL._PHI), OL._DD, axis=0)) / (OL.R_EARTH * np.cos(OL._PHI))
                z5 = OL._wconv(OL.uniform_filter(OL._HS * z, size=3, mode=("nearest", "wrap")), OL._K5) / OL._DEN5
                Z.append(z5[OL._PJ, OL._PK] * 1e5)
            Z = np.array(Z); share += Z >= LOCK; mean += Z; n_ok += 1
        except Exception as e:
            print(f"member {m:02d} skipped ({type(e).__name__})")
    if not n_ok:
        return False
    share /= n_ok; mean /= n_ok
    live = np.zeros(len(pts), bool)                 # same points as the outlook: no active storms, ocean, gate open
    tj = os.path.join(outdir, f"terms_{day:%Y%m%d}.json")
    if os.path.exists(tj):
        T = json.load(open(tj)); idx = {(round(a, 1), round(b, 1)): i for i, (a, b) in enumerate(pts)}
        live[[idx[(round(a, 1), round(b, 1))] for a, b in zip(T["lat"], T["lon"])]] = True
    else:
        live[:] = True
    keep = np.where((share.max(0) > 0) & live)[0]
    _w(os.path.join(outdir, f"ensemble_{day:%Y%m%d}.json"),
       dict(issued=f"{day:%Y-%m-%d} 00 UTC", members=n_ok, leads=ELEADS, threshold_1e5=LOCK,
            lat=[float(x) for x in pts[keep, 0]], lon=[float(x) for x in pts[keep, 1]],
            share=[[round(float(x), 2) for x in r[keep]] for r in share], mean_spin=[[round(float(x), 2) for x in r[keep]] for r in mean]))
    return True


def make(outdir, members=10, ensemble=True):
    days = sorted(f[6:14] for f in os.listdir(outdir) if f.startswith("terms_") and f.endswith(".json"))
    for d in days:
        if not os.path.exists(os.path.join(outdir, f"leads_{d}.json")):
            leads_view(outdir, datetime.strptime(d, "%Y%m%d").replace(tzinfo=timezone.utc))
    if ensemble and days and members > 0:
        last = datetime.strptime(days[-1], "%Y%m%d").replace(tzinfo=timezone.utc)
        if not os.path.exists(os.path.join(outdir, f"ensemble_{days[-1]}.json")):
            ensemble_view(outdir, last, members)
    _w(os.path.join(outdir, "views.json"), dict(leads=sorted(f[6:14] for f in os.listdir(outdir) if f.startswith("leads_")),
                                              ensemble=sorted(f[9:17] for f in os.listdir(outdir) if f.startswith("ensemble_"))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--members", type=int, default=10)
    a = ap.parse_args(); make(a.out, a.members)
