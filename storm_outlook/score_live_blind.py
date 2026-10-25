"""Frozen scorer for the live blind test, 26 Oct - 31 Dec 2026 (LIVE_BLIND_2026_PREREG.md).
Run once, after the frozen app has produced every day of the period and IBTrACS covers it:
    python score_live_blind.py --out <folder produced by run_local.py> --storms <IBTrACS-format CSV>
"""
import argparse, json, os, sys
from datetime import datetime, timedelta
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify as VF, bomb as BM, terms as TM

D0, D1 = datetime(2026, 10, 26), datetime(2026, 12, 31)
DAYS = [D0 + timedelta(days=i) for i in range((D1 - D0).days + 1)]


def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y, bool); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None


def genesis(storms_csv):
    import pandas as pd
    I = pd.read_csv(storms_csv, skiprows=[1], low_memory=False, keep_default_na=False, na_values=["", " "])
    for c in ("LAT", "LON", "USA_WIND", "WMO_WIND"): I[c] = pd.to_numeric(I[c], errors="coerce")
    I["t"] = pd.to_datetime(I.ISO_TIME, errors="coerce"); I = I[(I.t.dt.hour % 6 == 0) & (I.t.dt.minute == 0)]
    I["W"] = I.USA_WIND.fillna(I.WMO_WIND); I["LON"] = I.LON % 360
    return I[I.W >= 34].sort_values("t").groupby("SID").first().reset_index()


def main(out, storms_csv):
    res = dict(period=[str(D0.date()), str(D1.date())])
    have = [d for d in DAYS if os.path.exists(os.path.join(out, f"terms_{d:%Y%m%d}.json"))]
    res["days_with_output"] = len(have); res["days_expected"] = len(DAYS)
    # T1: the app as deployed (verify.py, unchanged)
    card, _, _ = VF.score(out, storms_csv)
    res["T1"] = dict(storms_alerted_24h=card.get("storms_alerted_24h"), storms_scored=card.get("storms_scored"),
                     became_storm_yellow=card.get("became_storm", {}).get("yellow"))
    res["T1_pass"] = bool((card.get("storms_alerted_24h") or 0) >= 0.70 and 0.25 <= (card.get("became_storm", {}).get("yellow") or 0) <= 0.55)
    # T2, T3: tropical formula vs trees at live points; lock-in shape
    G = genesis(storms_csv); gt = G.t.values
    Y, ST, SF, SP, SH = [], [], [], [], []
    for d in have:
        T = json.load(open(os.path.join(out, f"terms_{d:%Y%m%d}.json")))
        la, lo = np.array(T["lat"]), np.array(T["lon"])
        m = (gt > np.datetime64(d)) & (gt <= np.datetime64(d + timedelta(hours=72)))
        g = G[m]
        y = (np.hypot(la[:, None] - g.LAT.values[None], ((lo[:, None] - g.LON.values[None] + 180) % 360 - 180) * np.cos(np.deg2rad(la[:, None]))).min(1) <= 5) if len(g) else np.zeros(len(la), bool)
        Y.append(y); ST.append(T["tree_chance"]); SF.append(T["formula_chance"]); SP.append(np.array(T["raw"]["spin_1e5"], float)); SH.append(np.array(T["raw"]["shear_ms"], float))
    Y, ST, SF, SP, SH = (np.concatenate(v) for v in (Y, ST, SF, SP, SH))
    res["T2"] = dict(points=int(len(Y)), storm_points=int(Y.sum()), AUC_trees=auc(Y, ST), AUC_formula=auc(Y, SF))
    res["T2_pass"] = bool((res["T2"]["AUC_trees"] or 0) >= 0.90 and (res["T2"]["AUC_formula"] or 0) >= 0.87)
    dec = np.digitize(SP, np.percentile(SP, np.arange(10, 100, 10))); ratios = []
    for k in range(10):
        hi, lo_ = (dec == k) & (SH >= 15), (dec == k) & (SH <= 8)
        ratios.append(float(Y[hi].mean() / Y[lo_].mean()) if hi.any() and lo_.any() and Y[lo_].mean() > 0 else None)
    top_hi_storms = int(Y[(dec == 9) & (SH >= 15)].sum())
    low9 = [r for r in ratios[:9] if r is not None]
    res["T3"] = dict(penalty_ratio_by_spin_decile=ratios, storm_points_top_decile_high_shear=top_hi_storms)
    res["T3_pass"] = bool(ratios[9] is not None and low9 and top_hi_storms >= 10 and ratios[9] >= 2 * float(np.median(low9)))
    # B1: explosive lows, trees and formula, all points
    ver = BM.verify(out, lambda p, o, **k: None, max_days=400)
    ev = [e for e in ver["events"] if not e["tropical_cyclone"]]
    Yb, Sb, Fb = [], [], []
    for d in DAYS:
        fa = os.path.join(out, "archive", f"bomb_features_{d:%Y%m%d}.npz")
        if not os.path.exists(fa): continue
        fx = np.load(fa)["X"].astype(np.float32)
        y = np.zeros(len(BM.BP), bool)
        for k in range(3):
            dk = (d + timedelta(days=k)).strftime("%Y%m%d"); E = [e for e in ev if e["start"] == dk]
            if E:
                la, lo = np.array([e["lat"] for e in E]), np.array([e["lon"] for e in E])
                y |= np.hypot(BM.BP[:, 0, None] - la[None], ((BM.BP[:, 1, None] - lo[None] + 180) % 360 - 180) * np.cos(np.deg2rad(BM.BP[:, 0, None]))).min(1) <= 5
        Yb.append(y); Sb.append(BM.chance(fx, d)); Fb.append(TM.bomb(fx, BM.BP)[1])
    if Yb:
        Yb, Sb, Fb = (np.concatenate(v) for v in (Yb, Sb, Fb))
        res["B1"] = dict(events=len(ev), points=int(len(Yb)), event_points=int(Yb.sum()), AUC_trees=auc(Yb, Sb), AUC_formula=auc(Yb, Fb))
        res["B1_pass"] = bool((res["B1"]["AUC_trees"] or 0) >= 0.90 and (res["B1"]["AUC_formula"] or 0) >= 0.78)
    json.dump(res, open(os.path.join(out, "live_blind_2026_score.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--storms", required=True)
    a = ap.parse_args(); main(a.out, a.storms)
