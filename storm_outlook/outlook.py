"""Storm formation outlook: daily tropical-cyclone formation outlook from the NOAA GEFS control forecast.

Usage:
    python outlook.py --date 2026-10-20 --out site            # live run for one 00 UTC issue
    python outlook.py --out site                               # today's 00 UTC issue

Licence: PolyForm Noncommercial 1.0.0 (see LICENSE.md). Experimental; not an official warning.
"""
import argparse, io, json, os, sys, time
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from scipy.ndimage import uniform_filter, maximum_filter, label
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "config.json")))
PTS = np.load(os.path.join(HERE, "model", "points.npy"))            # (2301, 2) lat, lon (0..360)
TREES = dict(np.load(os.path.join(HERE, "model", "trees.npz")))
LEADS = (3, 24, 48, 96, 120)
FEATURES = [f"{n}_{h}" for h in LEADS for n in ("z5", "zmax", "sh", "reach", "pw", "q7")] + ["abslat", "scos", "ssin"]
R_EARTH = 6.371e6
SES = requests.Session()


def _write_json(path, obj, **kw):
    """Write atomically: a crash mid-write never leaves a half-written file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, **kw)
    os.replace(tmp, path)


# ---------------------------------------------------------------- data access
def get(url, headers=None, tries=6):
    for k in range(tries):
        try:
            r = SES.get(url, headers=headers, timeout=120)
            r.raise_for_status()
            return r
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(4 * (k + 1))


def grib_field(stem, var, level, lead):
    """One GRIB2 message by HTTP byte range, regridded to 0.5 deg, rows 45S..45N (south first)."""
    import eccodes
    idx = [l.split(":") for l in get(stem + ".idx").text.splitlines()]
    k = [i for i, a in enumerate(idx) if a[3] == var and a[4] == level and a[5] == f"{lead} hour fcst"][0]
    a = int(idx[k][1])
    rng = f"bytes={a}-{int(idx[k + 1][1]) - 1}" if k + 1 < len(idx) else f"bytes={a}-"
    g = eccodes.codes_new_from_message(get(stem, {"Range": rng}).content)
    nj, ni = eccodes.codes_get(g, "Nj"), eccodes.codes_get(g, "Ni")
    v = eccodes.codes_get_values(g).reshape(nj, ni)
    eccodes.codes_release(g)
    if ni == 1440:
        v = v[::2, ::2]
    return v[90:271][::-1].astype(np.float64)


FIELDS = [("a", "UGRD", "850 mb", "u850"), ("a", "VGRD", "850 mb", "v850"), ("a", "UGRD", "200 mb", "u200"),
          ("a", "VGRD", "200 mb", "v200"), ("b", "CAPE", "surface", "cape"),
          ("a", "PWAT", "entire atmosphere (considered as a single layer)", "pwat"), ("b", "SPFH", "700 mb", "q700")]


def fetch_gefs(day):
    base = CFG["sources"]["gefs"]
    out = {}
    for h in LEADS:
        for prod_key, var, lev, name in FIELDS:
            prod = "pgrb2ap5" if prod_key == "a" else "pgrb2bp5"
            stem = f"{base}/gefs.{day:%Y%m%d}/00/atmos/{prod}/gec00.t00z.{prod[:6]}.0p50.f{h:03d}"
            out[(name, h)] = grib_field(stem, var, lev, h)
    return out


def parse_tcvitals(text):
    systems = []
    for line in text.splitlines():
        p = line.split()
        if len(p) < 13:
            continue
        try:
            lat = int(p[5][:-1]) / 10 * (1 if p[5][-1] == "N" else -1)
            lon = int(p[6][:-1]) / 10 * (1 if p[6][-1] == "E" else -1)
            kt = float(p[12]) / 0.514444
        except ValueError:
            continue
        systems.append(dict(agency=p[0], id=p[1], name=p[2], lat=lat, lon=lon % 360, kt=kt))
    return systems


def fetch_storms(day):
    """Active systems at 00 UTC: NCEP TC vitals (NHC + JTWC, real time); fallback IBTrACS ACTIVE (lagged ~2-3 days)."""
    url = f"{CFG['sources']['tcvitals']}/gfs.{day:%Y%m%d}/00/atmos/gfs.t00z.syndata.tcvitals.tm00"
    try:
        r = SES.get(url, timeout=60)
        if r.status_code == 200:
            return parse_tcvitals(r.text), "tcvitals"
    except Exception:
        pass
    try:  # fallback: latest IBTrACS provisional fix of each storm within the last 3 days, no extrapolation
        import csv
        text = get(CFG["sources"]["ibtracs_active"]).text.splitlines()
        rows = list(csv.DictReader(text[:1] + text[2:]))
        latest = {}
        for r in rows:
            t = datetime.strptime(r["ISO_TIME"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            if day - timedelta(days=3) <= t <= day and (r["SID"] not in latest or t > latest[r["SID"]][0]):
                latest[r["SID"]] = (t, r)
        out = []
        for t, r in latest.values():
            w = r["USA_WIND"].strip() or r["WMO_WIND"].strip()
            if w:
                out.append(dict(agency="IBTrACS", id=r["SID"], name=r["NAME"], lat=float(r["LAT"]), lon=float(r["LON"]) % 360, kt=float(w)))
        return out, "ibtracs_active_lagged"
    except Exception:
        return [], "none"


# ---------------------------------------------------------------- features
def _kernel(res, rmax):
    n = int(np.ceil(rmax / res))
    y, x = np.mgrid[-n:n + 1, -n:n + 1] * res
    return (np.hypot(y, x * np.cos(np.deg2rad(20))) <= rmax).astype(float)


def _wconv(f, k):
    n = k.shape[1] // 2
    fp = np.concatenate([f[:, -n:], f, f[:, :n]], 1)
    return fftconvolve(fp, k, mode="same")[:, n:-n]


def _wmax(f, fp):
    n = fp.shape[1] // 2
    sp = np.concatenate([f[:, -n:], f, f[:, :n]], 1)
    return maximum_filter(sp, footprint=fp, mode="nearest")[:, n:-n]


_LAT = np.arange(-45, 45.01, 0.5)
_PHI = np.deg2rad(_LAT)[:, None]
_DD = np.deg2rad(0.5)
_HS = np.sign(_LAT)[:, None]
_K5, _K12 = _kernel(0.5, 5), _kernel(0.5, 12)
_DEN5, _DEN12 = _wconv(np.ones((len(_LAT), 720)), _K5), _wconv(np.ones((len(_LAT), 720)), _K12)
_PJ = np.round((PTS[:, 0] + 45) / 0.5).astype(int)
_PK = np.round(PTS[:, 1] / 0.5).astype(int) % 720


def features(F):
    """30 forecast features x 2301 points (same order as training)."""
    rows = []
    for h in LEADS:
        u, v, u2, v2 = F[("u850", h)], F[("v850", h)], F[("u200", h)], F[("v200", h)]
        z = ((np.roll(v, -1, 1) - np.roll(v, 1, 1)) / (2 * _DD) - np.gradient(u * np.cos(_PHI), _DD, axis=0)) / (R_EARTH * np.cos(_PHI))
        z5 = _wconv(uniform_filter(_HS * z, size=3, mode=("nearest", "wrap")), _K5) / _DEN5
        rows += [z5[_PJ, _PK], _wmax(z5, _K5)[_PJ, _PK], (_wconv(np.hypot(u2 - u, v2 - v), _K5) / _DEN5)[_PJ, _PK],
                 (_wconv((np.nan_to_num(F[("cape", h)]) > 1000).astype(float), _K12) / _DEN12)[_PJ, _PK],
                 (_wconv(F[("pwat", h)], _K5) / _DEN5)[_PJ, _PK], (_wconv(F[("q700", h)], _K5) / _DEN5)[_PJ, _PK]]
    return np.stack(rows).astype(np.float32)


# ---------------------------------------------------------------- model
def _trees_raw(target, X):
    d = {k[len(target) + 1:]: v for k, v in TREES.items() if k.startswith(target + "_")}
    out = np.full(len(X), d["baseline"][0])
    rows = np.arange(len(X))
    for r in d["roots"]:
        node = np.full(len(X), r)
        while True:
            leaf = d["leaf"][node].astype(bool)
            if leaf.all():
                break
            x = X[rows, np.where(leaf, 0, d["feature"][node])]
            left = np.where(np.isnan(x), d["miss_left"][node].astype(bool), x <= d["thr"][node])
            node = np.where(leaf, node, np.where(left, d["left"][node], d["right"][node]) + r)
        out += d["value"][node]
    return out


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _expit(z):
    return 1 / (1 + np.exp(-z))


def deg_dist(la1, lo1, la2, lo2):
    return np.hypot(la1[:, None] - la2[None], ((lo1[:, None] - lo2[None] + 180) % 360 - 180) * np.cos(np.deg2rad(la1[:, None])))


def probabilities(feat, day, storms):
    """Calibrated chance of formation within 5 deg: days 0-3 (map), days 3-5, and alert-point versions."""
    doy = day.timetuple().tm_yday
    X = np.c_[feat.T.astype(np.float64), np.abs(PTS[:, 0]), np.full(len(PTS), np.cos(2 * np.pi * doy / 365.25)),
              np.full(len(PTS), np.sin(2 * np.pi * doy / 365.25))]
    z03 = CFG["platt"]["y03"][0] + CFG["platt"]["y03"][1] * _logit(_expit(_trees_raw("y03", X)))
    z45 = CFG["platt"]["y45"][0] + CFG["platt"]["y45"][1] * _logit(_expit(_trees_raw("y45", X)))
    sys34 = [s for s in storms if s["kt"] >= CFG["storm_min_kt"]]
    dmin = (deg_dist(PTS[:, 0], PTS[:, 1], np.array([s["lat"] for s in sys34]), np.array([s["lon"] for s in sys34])).min(1)
            if sys34 else np.full(len(PTS), 99.0))
    live = (feat[FEATURES.index("reach_3")] > CFG["gate_reach"]) & (dmin > CFG["storm_mask_deg"])
    off = np.zeros(len(PTS))
    for (a, b), o in zip(CFG["neighbour_offsets"]["bands"], CFG["neighbour_offsets"]["logodds"]):
        off += o * ((dmin > a) & (dmin <= b))
    p03_map = np.where(live, _expit(z03 + off), 0.0)
    p03, p45 = np.where(live, _expit(z03), 0.0), np.where(live, _expit(z45), 0.0)
    keep = dmin > CFG["alert_mask_deg"]
    wp = CFG["west_poleward"]
    for s in storms:
        if s["kt"] < wp["min_kt"]:
            continue
        hs = np.sign(s["lat"]) or 1.0
        dy = (PTS[:, 0] - s["lat"]) * hs * 111.2
        dx = ((PTS[:, 1] - s["lon"] + 180) % 360 - 180) * 111.2 * np.cos(np.deg2rad((PTS[:, 0] + s["lat"]) / 2))
        r = np.hypot(dx, dy)
        keep &= ~((r >= wp["r_min_km"]) & (r <= wp["r_max_km"]) & (dx < 0) & (dy > 0))
    return p03_map, np.where(keep, p03, 0.0), np.where(keep, p45, 0.0)


# ---------------------------------------------------------------- regions and cluster rule
_LAT_U = np.unique(PTS[:, 0])
_LON_U = np.arange(0, 360, 2.5)


def regions(a03, a45, day, prev):
    """Connected alert regions for one day; age carried from yesterday's regions (prev) by 5-deg overlap."""
    ap = CFG["alert_point"]
    on = (a03 >= ap["p03"] * ap["scale"]) | (a45 >= ap["p45"] * ap["scale"])
    gi, gj = np.searchsorted(_LAT_U, PTS[:, 0]), np.round(PTS[:, 1] / 2.5).astype(int)
    G = np.zeros((len(_LAT_U), len(_LON_U)), bool); G[gi, gj] = on
    P3 = np.zeros(G.shape); P4 = np.zeros(G.shape); P3[gi, gj] = a03; P4[gi, gj] = a45
    lab, n = label(G, structure=np.ones((3, 3)))
    for a, b in zip(lab[:, 0], lab[:, -1]):           # longitude wrap
        if a and b and a != b:
            lab[lab == b] = a
    consecutive = prev is not None and prev.get("date") == (day - timedelta(days=1)).strftime("%Y-%m-%d")
    out = []
    for k in np.unique(lab[lab > 0]):
        jj, kk = np.where(lab == k)
        r = dict(lat=_LAT_U[jj].tolist(), lon=_LON_U[kk].tolist(), size=int(len(jj)), peak03=float(P3[jj, kk].max()),
                 peak45=float(P4[jj, kk].max()), age=1)
        if consecutive:
            for q in prev["regions"]:
                if deg_dist(np.array(r["lat"]), np.array(r["lon"]), np.array(q["lat"]), np.array(q["lon"])).min() <= CFG["cluster"]["link_deg"]:
                    r["age"] = max(r["age"], q["age"] + 1)
        out.append(r)
    return out


def level(r):
    c, L = CFG["cluster"], CFG["levels"]
    if r["size"] < c["min_points"] or r["age"] < c["min_age_days"]:
        return None
    if r["size"] >= L["red"]["min_points"] and r["age"] >= L["red"]["min_age_days"] and r["peak03"] >= L["red"]["min_peak_p03"]:
        return "red"
    if r["size"] >= L["orange"]["min_points"] and r["age"] >= L["orange"]["min_age_days"]:
        return "orange"
    return "yellow"


# ---------------------------------------------------------------- outputs
def geojson(day, regs, storms, p03_map, source):
    feats = []
    for r in regs:
        lv = level(r)
        if lv is None:
            continue
        # shift the cell centre to -180..180 first, then build corners, so a cell on the date line keeps its 2.5-degree width
        cells = []
        for la, lo in zip(r["lat"], r["lon"]):
            c = lo - 360 if lo > 180 else lo
            cells.append([[c - 1.25, la - 1.25], [c + 1.25, la - 1.25], [c + 1.25, la + 1.25], [c - 1.25, la + 1.25], [c - 1.25, la - 1.25]])
        feats.append(dict(type="Feature", geometry=dict(type="MultiPolygon", coordinates=[[c] for c in cells]),
                          properties=dict(kind="alert", level=lv, size=r["size"], age_days=r["age"],
                                          peak_chance_0_3d=round(r["peak03"], 3), peak_chance_3_5d=round(r["peak45"], 3))))
    for s in storms:
        feats.append(dict(type="Feature", geometry=dict(type="Point", coordinates=[(s["lon"] - 360 if s["lon"] > 180 else s["lon"]), s["lat"]]),
                          properties=dict(kind="active_storm", name=s["name"], id=s["id"], wind_kt=round(s["kt"]))))
    for (la, lo), p in zip(PTS, p03_map):
        if p >= 0.01:
            feats.append(dict(type="Feature", geometry=dict(type="Point", coordinates=[(lo - 360 if lo > 180 else lo), la]),
                              properties=dict(kind="chance_0_3d", value=round(float(p), 3))))
    return dict(type="FeatureCollection", issued=f"{day:%Y-%m-%d} 00 UTC", storm_source=source, model=CFG["version"],
                note="Experimental research outlook. Not an official warning; follow your national meteorological service.",
                features=feats)


def run(day, outdir, feat=None, storms=None):
    os.makedirs(outdir, exist_ok=True)
    state_path = os.path.join(outdir, "state.json")
    prev = json.load(open(state_path)) if os.path.exists(state_path) else None
    if prev is not None and prev["date"] >= f"{day:%Y-%m-%d}":
        print(f"{day:%Y-%m-%d}: already done (state is at {prev['date']}); skipped")
        return None
    if storms is None:
        storms, source = fetch_storms(day)
    else:
        source = "supplied"
    if feat is None:
        feat = features(fetch_gefs(day))
    p03_map, a03, a45 = probabilities(feat, day, storms)
    regs = regions(a03, a45, day, prev)
    gj = geojson(day, regs, storms, p03_map, source)
    for name in (f"outlook_{day:%Y%m%d}.geojson", "outlook.geojson"):
        _write_json(os.path.join(outdir, name), gj, separators=(",", ":"))
    os.makedirs(os.path.join(outdir, "archive"), exist_ok=True)   # inputs kept for the yearly reviewed refit
    fa = os.path.join(outdir, "archive", f"features_{day:%Y%m%d}")
    np.savez_compressed(fa + ".tmp.npz", X=feat.astype(np.float32),
                        storms=np.array([[s["lat"], s["lon"], s["kt"]] for s in storms], dtype=np.float32).reshape(-1, 3))
    os.replace(fa + ".tmp.npz", fa + ".npz")
    # state is written last: its date is the single record of the last completed day (used to resume)
    _write_json(state_path, dict(date=f"{day:%Y-%m-%d}", regions=regs), separators=(",", ":"))
    for f in os.listdir(outdir):                      # keep 400 days of daily outlooks (verification needs a season)
        if f.startswith("outlook_") and f.endswith(".geojson"):
            try:
                if (day - datetime.strptime(f[8:16], "%Y%m%d").replace(tzinfo=timezone.utc)).days > 400:
                    os.remove(os.path.join(outdir, f))
            except ValueError:
                pass
    n = {lv: sum(level(r) == lv for r in regs) for lv in ("yellow", "orange", "red")}
    print(f"{day:%Y-%m-%d}: storms from {source} ({len(storms)}); alert regions {n}")
    return gj


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="issue date YYYY-MM-DD (00 UTC); default today UTC")
    ap.add_argument("--out", default="site")
    a = ap.parse_args()
    d = (datetime.strptime(a.date, "%Y-%m-%d") if a.date else datetime.now(timezone.utc)).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    sp = os.path.join(a.out, "state.json")
    last = json.load(open(sp))["date"] if os.path.exists(sp) else None
    if last != (d - timedelta(days=1)).strftime("%Y-%m-%d"):
        # first run or a gap: rebuild region ages from the previous 4 days (the cluster rule needs >= 3 days)
        if os.path.exists(sp):
            os.remove(sp)
        for k in range(4, 0, -1):
            try:
                run(d - timedelta(days=k), a.out)
            except Exception as e:
                print(f"warm-up day {d - timedelta(days=k):%Y-%m-%d} skipped: {type(e).__name__}")
    run(d, a.out)
