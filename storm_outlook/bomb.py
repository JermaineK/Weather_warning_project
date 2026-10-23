"""Explosive mid-latitude cyclone ("bomb") outlook layer. EXPLOSIVE_CYCLONE_PREREG.md; blind-tested on 2000-09.

Alert point: a 2.5-deg point at 25-70 deg latitude where the model's chance of an explosive-deepening onset within 5 deg
in the next 3 days exceeds the frozen threshold. Events are detected from the forecast system's own 3-h pressure field on
following days: a tracked low whose central pressure falls >= 24 hPa x sin|lat| / sin 60 deg in 24 h.
Licence: PolyForm Noncommercial 1.0.0.
"""
import json, os
from datetime import timedelta
import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter, uniform_filter

HERE = os.path.dirname(os.path.abspath(__file__))
BCFG = json.load(open(os.path.join(HERE, "config.json")))["bomb"]
BP = np.load(os.path.join(HERE, "model", "bomb_points.npy"))
BTREES = dict(np.load(os.path.join(HERE, "model", "bomb_trees.npz")))
LAT = np.arange(90, -90.01, -1.0); LON = np.arange(0, 360, 1.0)
_bi = np.round(90 - BP[:, 0]).astype(int); _bk = np.round(BP[:, 1]).astype(int) % 360
_R, _G, _OM = 6.371e6, 9.80665, 7.292e-5
_phi = np.deg2rad(LAT)[:, None]; _f = 2 * _OM * np.sin(_phi); _hs = np.sign(LAT)[:, None]
# (GEFS product, variable, level, leads)
LIVE_FIELDS = [("b", "PRES", "mean sea level", (3, 24, 48, 72)), ("a", "HGT", "500 mb", (24, 48, 72)), ("a", "TMP", "850 mb", (24, 48, 72)),
               ("a", "UGRD", "250 mb", (24, 48, 72)), ("a", "VGRD", "250 mb", (24, 48, 72)),
               ("a", "PWAT", "entire atmosphere (considered as a single layer)", (48,))]


def _wrap(func, a, size):
    n = size[1] // 2
    ap = np.concatenate([a[:, -n:], a, a[:, :n]], 1)
    return func(ap, size=size, mode="nearest")[:, n:-n]


def lows(p3):
    """Local minima of the 3-h MSLP (hPa) on the 1-deg grid: 9x9 window, < 1010 hPa, |lat| 20-75. Returns (n, 3) lat, lon, hPa."""
    mn = _wrap(minimum_filter, p3, (9, 9))
    ii, kk = np.where((p3 == mn) & (p3 < 1010) & (np.abs(LAT)[:, None] >= 20) & (np.abs(LAT)[:, None] <= 75))
    return np.c_[LAT[ii], LON[kk], p3[ii, kk]]


def features(F):
    """17 features x 5472 points from 1-deg fields F[(name, lead)]: name in mslp (hPa), z500, t850, u250, v250, pwat."""
    P = {h: F[("mslp", h)] for h in (3, 24, 48, 72)}
    Pm = {h: _wrap(minimum_filter, P[h], (11, 15)) for h in P}
    norm = np.sin(np.deg2rad(60)) / np.maximum(np.abs(np.sin(_phi)), np.sin(np.deg2rad(20)))
    out = [Pm[h] - 1000 for h in (3, 24, 48, 72)]
    out += [(Pm[h] - Pm[h - 24 if h > 24 else 3]) * norm for h in (24, 48, 72)]
    dl = np.deg2rad(1.0)
    for h in (24, 48, 72):
        Z = F[("z500", h)]
        lap = ((np.roll(Z, -1, 1) - 2 * Z + np.roll(Z, 1, 1)) / (dl ** 2 * np.cos(_phi) ** 2) +
               np.gradient(np.gradient(Z, -dl, axis=0) * np.cos(_phi), -dl, axis=0) / np.cos(_phi)) / _R ** 2
        out.append(_wrap(uniform_filter, np.nan_to_num(_hs * _G / np.where(np.abs(_f) < 1e-5, np.inf, _f) * lap), (5, 7)) * 1e5)
    for h in (24, 48, 72):
        T = F[("t850", h)]
        gx = (np.roll(T, -1, 1) - np.roll(T, 1, 1)) / (2 * dl * _R * np.cos(_phi)); gy = np.gradient(T, -dl, axis=0) / _R
        out.append(_wrap(uniform_filter, np.hypot(gx, gy), (11, 15)) * 1e5)
    for h in (24, 48, 72):
        out.append(_wrap(maximum_filter, np.hypot(F[("u250", h)], F[("v250", h)]), (11, 15)))
    out.append(_wrap(uniform_filter, F[("pwat", 48)], (11, 15)))
    return np.stack([o[_bi, _bk] for o in out]).astype(np.float16).astype(np.float64)   # float16 as in training


def chance(feat, day):
    doy = day.timetuple().tm_yday
    ph = 2 * np.pi * (doy - np.where(BP[:, 0] < 0, 182.6, 0)) / 365.25
    X = np.c_[feat.T, np.abs(BP[:, 0]), np.cos(ph), np.sin(ph)]
    out = np.full(len(X), BTREES["baseline"][0]); rows = np.arange(len(X))
    for r in BTREES["roots"]:
        node = np.full(len(X), r)
        while True:
            leaf = BTREES["leaf"][node].astype(bool)
            if leaf.all():
                break
            x = X[rows, np.where(leaf, 0, BTREES["feature"][node])]
            left = np.where(np.isnan(x), BTREES["miss_left"][node].astype(bool), x <= BTREES["thr"][node])
            node = np.where(leaf, node, np.where(left, BTREES["left"][node], BTREES["right"][node]) + r)
        out += BTREES["value"][node]
    return 1 / (1 + np.exp(-out))


def fetch_live(day, get, gefs_base):
    F = {}
    for prod, var, lev, leads in LIVE_FIELDS:
        p = "pgrb2ap5" if prod == "a" else "pgrb2bp5"
        for h in leads:
            stem = f"{gefs_base}/gefs.{day:%Y%m%d}/00/atmos/{p}/gec00.t00z.{p[:6]}.0p50.f{h:03d}"
            L = [l.split(":") for l in get(stem + ".idx").text.splitlines()]
            k = [i for i, a in enumerate(L) if a[3] == var and a[4] == lev and a[5] == f"{h} hour fcst"][0]
            rng = f"bytes={L[k][1]}-{int(L[k + 1][1]) - 1}" if k + 1 < len(L) else f"bytes={L[k][1]}-"
            import eccodes
            g = eccodes.codes_new_from_message(get(stem, headers={"Range": rng}).content)
            nj, ni = eccodes.codes_get(g, "Nj"), eccodes.codes_get(g, "Ni")
            v = eccodes.codes_get_values(g).reshape(nj, ni); eccodes.codes_release(g)
            v = v[::ni // 360, ::ni // 360].astype(np.float64)
            name = {"PRES": "mslp", "HGT": "z500", "TMP": "t850", "UGRD": "u250", "VGRD": "v250", "PWAT": "pwat"}[var]
            F[(name, h)] = v / 100.0 if name == "mslp" else v
    return F


def run(day, outdir, get, gefs_base, write_json, storms=()):
    F = fetch_live(day, get, gefs_base)
    p = chance(features(F), day)
    on = p >= BCFG["threshold"]
    L = lows(F[("mslp", 3)])
    write_json(os.path.join(outdir, f"bomb_{day:%Y%m%d}.json"),
               dict(issued=f"{day:%Y-%m-%d} 00 UTC", threshold=BCFG["threshold"],
                    alerts=[[float(a), float(b), round(float(c), 3)] for (a, b), c in zip(BP[on], p[on])],
                    lows=[[float(a), float(b), round(float(c), 1)] for a, b, c in L],
                    tropical=[[float(s["lat"]), float(s["lon"])] for s in storms if s["kt"] >= 34]), separators=(",", ":"))
    return int(on.sum()), len(L)


def _gc_km(la1, lo1, la2, lo2):
    p1, p2 = np.deg2rad(la1), np.deg2rad(la2)
    return 6371 * np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(np.deg2rad(lo2 - lo1)), -1, 1))


def verify(outdir, write_json, max_days=60):
    """Detect explosive deepening in the saved daily lows and match to alerts issued 1-2 days before the 24-h window starts."""
    from datetime import datetime
    files = sorted(f for f in os.listdir(outdir) if f.startswith("bomb_") and f.endswith(".json") and f[5:13].isdigit())[-max_days:]
    B = {f[5:13]: json.load(open(os.path.join(outdir, f))) for f in files}
    D = sorted(B); events = []; track = {}; nxt = 0; started = set()
    for d0 in D:
        d1 = (datetime.strptime(d0, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
        if d1 not in B: continue
        A, C = np.array(B[d0]["lows"]).reshape(-1, 3), np.array(B[d1]["lows"]).reshape(-1, 3)
        if not len(A) or not len(C): continue
        Dm = _gc_km(A[:, 0][:, None], A[:, 1][:, None], C[:, 0][None], C[:, 1][None]); ui, uj = set(), set()
        for flat in np.argsort(Dm, axis=None):
            i, j = np.unravel_index(flat, Dm.shape)
            if Dm[i, j] > 1200: break
            if i in ui or j in uj: continue
            ui.add(i); uj.add(j)
            tid = track.get((d0, i))
            if tid is None: tid = nxt; nxt += 1; track[(d0, i)] = tid
            track[(d1, j)] = tid
            fall = (A[i, 2] - C[j, 2]) * np.sin(np.deg2rad(60)) / max(abs(np.sin(np.deg2rad((A[i, 0] + C[j, 0]) / 2))), 1e-3)
            if fall >= 24 and tid not in started and 25 <= abs(C[j, 0]) <= 70:
                started.add(tid); ev = dict(start=d0, lat=float(C[j, 0]), lon=float(C[j, 1]), fall_hpa=round(float(A[i, 2] - C[j, 2]), 1), p_end=round(float(C[j, 2]), 1))
                lead = 0
                for k in (1, 2):
                    dk = (datetime.strptime(d0, "%Y%m%d") - timedelta(days=k)).strftime("%Y%m%d")
                    al = np.array(B.get(dk, {}).get("alerts", [])).reshape(-1, 3)
                    if len(al) and np.min(np.hypot(al[:, 0] - ev["lat"], ((al[:, 1] - ev["lon"] + 180) % 360 - 180) * np.cos(np.deg2rad(ev["lat"])))) <= 5:
                        lead = max(lead, k)
                ev["alerted_days_before"] = lead
                ev["scorable"] = any((datetime.strptime(d0, "%Y%m%d") - timedelta(days=k)).strftime("%Y%m%d") in B for k in (1, 2))
                tc = np.array(B[d0].get("tropical", []) + B[d1].get("tropical", [])).reshape(-1, 2)   # a tropical cyclone, not a mid-latitude bomb
                ev["tropical_cyclone"] = bool(len(tc) and np.min(np.hypot(tc[:, 0] - ev["lat"], ((tc[:, 1] - ev["lon"] + 180) % 360 - 180) * np.cos(np.deg2rad(ev["lat"])))) <= 5)
                events.append(ev)
    res = dict(period=[D[0], D[-1]] if D else None, events=events)
    write_json(os.path.join(outdir, "bomb_verification.json"), res)
    return res
