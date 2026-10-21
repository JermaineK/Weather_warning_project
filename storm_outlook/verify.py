"""Automatic verification: scores past outlooks against confirmed storms ("hits", misses and false alarms).

Run after outlook.py (the workflow does this daily). It never changes the model; it writes a scorecard
and raises a drift flag if live performance leaves the range seen in the 2017-19 verification.

    python verify.py --out site                      # uses IBTrACS ACTIVE (provisional, 6-hourly)
    python verify.py --out site --storms local.csv   # any IBTrACS-format CSV
Licence: PolyForm Noncommercial 1.0.0.
"""
import argparse, csv, glob, io, json, os
from datetime import datetime, timedelta, timezone

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outlook import CFG, PTS, deg_dist, get

LEVELS = ("yellow", "orange", "red")
RANK = {lv: i for i, lv in enumerate(LEVELS)}


def load_confirmed(outdir, source=None):
    """Append newly confirmed storms (first fix >= 34 kt) to storms_confirmed.csv and return all of them."""
    path = os.path.join(outdir, "storms_confirmed.csv")
    known = {}
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            known[r["sid"]] = r
    text = open(source).read() if source else get(CFG["sources"]["ibtracs_active"]).text
    lines = text.splitlines()
    for r in csv.DictReader(lines[:1] + lines[2:]):
        w = (r.get("USA_WIND") or "").strip() or (r.get("WMO_WIND") or "").strip()
        try:
            w = float(w)
        except ValueError:
            continue
        if w < CFG["storm_min_kt"]:
            continue
        t = r["ISO_TIME"][:16]
        if r["SID"] not in known or t < known[r["SID"]]["genesis_utc"]:
            known[r["SID"]] = dict(sid=r["SID"], name=r.get("NAME", ""), genesis_utc=t, lat=r["LAT"], lon=str(float(r["LON"]) % 360))
    rows = sorted(known.values(), key=lambda r: r["genesis_utc"])
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["sid", "name", "genesis_utc", "lat", "lon"])
        wr.writeheader(); wr.writerows(rows)
    return rows


def load_outlooks(outdir):
    """Alert regions per issue day, as cell-centre arrays, from the archived daily GeoJSON files."""
    out = {}
    for f in sorted(glob.glob(os.path.join(outdir, "outlook_????????.geojson"))):
        day = datetime.strptime(os.path.basename(f)[8:16], "%Y%m%d").replace(tzinfo=timezone.utc)
        regs = []
        for ft in json.load(open(f))["features"]:
            p = ft["properties"]
            if p.get("kind") != "alert":
                continue
            cells = np.array([np.mean(np.array(poly[0][:4]), axis=0) for poly in ft["geometry"]["coordinates"]])
            regs.append(dict(level=p["level"], lat=cells[:, 1], lon=cells[:, 0] % 360))
        out[day] = regs
    return out


def score(outdir, storms_csv=None, lag_days=7):
    storms = load_confirmed(outdir, storms_csv)
    O = load_outlooks(outdir)
    if not O:
        return None
    first, last = min(O), max(O)
    st = []
    for s in storms:
        t = datetime.strptime(s["genesis_utc"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        la, lo = float(s["lat"]), float(s["lon"])
        if deg_dist(np.array([la]), np.array([lo]), PTS[:, 0], PTS[:, 1]).min() <= 5:  # inside the outlook domain
            st.append((s["sid"], s["name"], t, la, lo))
    newest = max([x[2] for x in st], default=first)
    complete_until = min(last, newest - timedelta(days=lag_days))  # outcomes treated as final up to here

    # storms: earliest lead at which any region (by level) lay within 5 deg of the genesis point
    srows = []
    for sid, name, t, la, lo in st:
        if not (first + timedelta(days=5) <= t <= last + timedelta(days=1)):
            continue
        lead = {lv: None for lv in LEVELS}
        for d in sorted(O):
            h = (t - d).total_seconds() / 3600
            if h <= 0 or h > 120:
                continue
            for r in O[d]:
                if deg_dist(np.array([la]), np.array([lo]), r["lat"], r["lon"]).min() <= 5:
                    for lv in LEVELS[: RANK[r["level"]] + 1]:
                        lead[lv] = max(lead[lv] or 0, h)
        srows.append(dict(sid=sid, name=name, genesis_utc=f"{t:%Y-%m-%d %H:%M}", **{f"lead_h_{lv}": lead[lv] for lv in LEVELS}))

    # region-days: did a storm form within 5 deg in the following 5 days?
    rrows = []
    for d, regs in sorted(O.items()):
        if d + timedelta(days=5) > complete_until:
            continue
        G = [(la, lo) for _, _, t, la, lo in st if d < t <= d + timedelta(days=5)]
        for r in regs:
            hit = bool(G) and deg_dist(r["lat"], r["lon"], np.array([g[0] for g in G]), np.array([g[1] for g in G])).min() <= 5
            rrows.append(dict(issue=f"{d:%Y-%m-%d}", level=r["level"], hit=hit))

    ref = CFG["verification_2017_2019"]
    caught = [r["lead_h_yellow"] is not None and r["lead_h_yellow"] >= 24 for r in srows]
    card = dict(updated=f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC", period=f"{first:%Y-%m-%d} to {last:%Y-%m-%d}",
                storms_scored=len(srows), storms_alerted_24h=(float(np.mean(caught)) if srows else None),
                region_days_scored=len(rrows),
                became_storm={lv: (float(np.mean([r["hit"] for r in rrows if RANK[r["level"]] >= RANK[lv]])) if any(RANK[r["level"]] >= RANK[lv] for r in rrows) else None) for lv in LEVELS},
                false_alarm_region_days=sum(not r["hit"] for r in rrows),
                reference_2017_2019=dict(storms_alerted_24h=ref["storms_alerted_24h"], became_storm=ref["became_storm"]))
    # drift flag: only once enough cases exist to tell drift from chance (binomial 2-sigma band)
    flags = []
    if len(srows) >= 20:
        p0 = ref["storms_alerted_24h"]; sd = (p0 * (1 - p0) / len(srows)) ** 0.5
        if abs(card["storms_alerted_24h"] - p0) > 2 * sd:
            flags.append(f"storms alerted >=24 h ahead {card['storms_alerted_24h']:.0%} vs reference {p0:.0%}")
    n_y = len(rrows) / 5   # a region is scored on several consecutive days with overlapping 5-day windows: ~5 region-days per independent case
    if n_y >= 40:
        p0 = ref["became_storm"]["yellow"]; sd = (p0 * (1 - p0) / n_y) ** 0.5
        if abs(card["became_storm"]["yellow"] - p0) > 2 * sd:
            flags.append(f"alert regions becoming storms {card['became_storm']['yellow']:.0%} vs reference {p0:.0%}")
    card["drift_flags"] = flags
    json.dump(card, open(os.path.join(outdir, "verification.json"), "w"), indent=1)
    with open(os.path.join(outdir, "verification_storms.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(srows[0]) if srows else ["sid"]); wr.writeheader(); wr.writerows(srows)
    print(json.dumps(card))
    return card, srows, rrows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site")
    ap.add_argument("--storms", help="local IBTrACS-format CSV instead of IBTrACS ACTIVE")
    a = ap.parse_args()
    score(a.out, a.storms)
