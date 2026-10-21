"""Yearly, human-reviewed refinement check. It never changes config.json by itself.

Uses the archived daily inputs (site/archive/features_*.npz) and confirmed storms (site/storms_confirmed.csv):
fits a candidate recalibration of the day 0-3 chances on the first part of the archive, scores it on the
most recent part, and writes refit_proposal.json. Adopt it only by pull request, after a pre-registered blind check.

    python refit_review.py --out site --holdout-days 120
Licence: PolyForm Noncommercial 1.0.0.
"""
import argparse, csv, glob, json, os
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.optimize import minimize

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outlook import CFG, FEATURES, PTS, _expit, _logit, _trees_raw, deg_dist


def build(outdir):
    storms = [(datetime.strptime(r["genesis_utc"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc), float(r["lat"]), float(r["lon"]))
              for r in csv.DictReader(open(os.path.join(outdir, "storms_confirmed.csv")))]
    last_known = max(s[0] for s in storms)
    Z, Y, D = [], [], []
    for f in sorted(glob.glob(os.path.join(outdir, "archive", "features_????????.npz"))):
        day = datetime.strptime(os.path.basename(f)[9:17], "%Y%m%d").replace(tzinfo=timezone.utc)
        if day + timedelta(days=3) > last_known - timedelta(days=7):
            continue                                            # outcome not yet final
        a = np.load(f); X = a["X"].astype(np.float64); st = a["storms"]
        doy = day.timetuple().tm_yday
        F = np.c_[X.T, np.abs(PTS[:, 0]), np.full(len(PTS), np.cos(2 * np.pi * doy / 365.25)), np.full(len(PTS), np.sin(2 * np.pi * doy / 365.25))]
        z = CFG["platt"]["y03"][0] + CFG["platt"]["y03"][1] * _logit(_expit(_trees_raw("y03", F)))
        s34 = st[st[:, 2] >= CFG["storm_min_kt"]] if len(st) else st
        dmin = deg_dist(PTS[:, 0], PTS[:, 1], s34[:, 0], s34[:, 1]).min(1) if len(s34) else np.full(len(PTS), 99.0)
        live = (X[FEATURES.index("reach_3")] > CFG["gate_reach"]) & (dmin > CFG["storm_mask_deg"])
        G = [(la, lo) for t, la, lo in storms if day < t <= day + timedelta(hours=72)]
        y = (deg_dist(PTS[:, 0], PTS[:, 1], np.array([g[0] for g in G]), np.array([g[1] for g in G])).min(1) <= 5) if G else np.zeros(len(PTS), bool)
        Z.append(z[live]); Y.append(y[live]); D.append(np.full(live.sum(), day.toordinal()))
    return np.concatenate(Z), np.concatenate(Y).astype(float), np.concatenate(D)


def nll(p, z, y):
    s = p[0] + p[1] * z
    return float(np.sum(np.logaddexp(0, s) - y * s))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site")
    ap.add_argument("--holdout-days", type=int, default=120)
    a = ap.parse_args()
    z, y, d = build(a.out)
    cut = d.max() - a.holdout_days
    tr, te = d <= cut, d > cut
    pen = lambda p: nll(p, z[tr], y[tr]) + p[0] ** 2 / (2 * 0.25) + (p[1] - 1) ** 2 / (2 * 0.0625)
    cand = minimize(pen, [0.0, 1.0], method="Nelder-Mead").x
    ll0, ll1 = nll([0.0, 1.0], z[te], y[te]), nll(cand, z[te], y[te])
    out = dict(created=f"{datetime.now(timezone.utc):%Y-%m-%d}", train_point_days=int(tr.sum()), test_point_days=int(te.sum()),
               test_storm_point_days=int(y[te].sum()), candidate_alpha_beta=[float(cand[0]), float(cand[1])],
               test_logloss_change_pct=100 * (ll1 / ll0 - 1),
               note=("Backtests (SELF_REFINE_RESULT.md) found this 2-parameter refit did not beat the frozen model. "
                     "Adopt only if the change is clearly negative over a full season and a pre-registered blind check agrees."))
    json.dump(out, open(os.path.join(a.out, "refit_proposal.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
