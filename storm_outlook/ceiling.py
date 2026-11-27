"""BETA, display only: each active storm's 'room to grow' = wind / potential intensity (PI).

PI is the Bister-Emanuel maximum potential intensity (tcpyPI), computed from GEFS control-run profiles
averaged over a 200-800 km ring around the storm, with sea temperature averaged over ocean within 400 km.
This follows PI_RATIO_RESULT.md (pre-registered, passed on unseen years), which used ERA5; the GEFS port
is checked against ERA5 in CEILING_PORT_CHECK.md. It never changes alerts or probabilities.

    import ceiling; ceiling.write(day, outdir)      # reads outlook_YYYYMMDD.geojson, writes ceiling_YYYYMMDD.json
Licence: PolyForm Noncommercial 1.0.0.
"""
import json, os
import numpy as np

LEVELS = [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50]
LAT = np.linspace(90, -90, 361); LON = np.arange(720) * 0.5
BANDS = [(0.40, "lots of room", "Far below its ceiling: storms like this usually strengthened next day."),
         (0.75, "some room", "Below its ceiling: strengthening still more likely than not."),
         (1.00, "near its ceiling", "Close to the most the sea and air can sustain: storms like this usually weakened."),
         (9e9, "above its ceiling", "Stronger than the local sea and air can sustain: weakening expected unless other energy (a trough) feeds it.")]


def _field(get, stem, idx, var, level):
    import eccodes
    k = [i for i, a in enumerate(idx) if a[3] == var and a[4] == level]
    if not k:
        return None
    k = k[0]; a = int(idx[k][1])
    rng = f"bytes={a}-{int(idx[k + 1][1]) - 1}" if k + 1 < len(idx) else f"bytes={a}-"
    g = eccodes.codes_new_from_message(get(stem, {"Range": rng}).content)
    nj, ni = eccodes.codes_get(g, "Nj"), eccodes.codes_get(g, "Ni")
    v = eccodes.codes_get_values(g).reshape(nj, ni); eccodes.codes_release(g)
    if ni == 1440:
        v = v[::2, ::2]
    return v.astype(np.float64)                       # 90N..90S, 0..359.5E at 0.5 deg


def fetch(get, base, ymd, cycle=0, lead=3):
    """T and RH profiles, mean sea-level pressure, surface temperature and land fraction, full globe."""
    F = {}
    for prod in ("pgrb2ap5", "pgrb2bp5"):
        stem = f"{base}/gefs.{ymd}/{cycle:02d}/atmos/{prod}/gec00.t{cycle:02d}z.{prod[:6]}.0p50.f{lead:03d}"
        idx = [l.split(":") for l in get(stem + ".idx").text.splitlines()]
        for lev in LEVELS:
            for var in ("TMP", "RH"):
                if (var, lev) not in F:
                    v = _field(get, stem, idx, var, f"{lev} mb")
                    if v is not None:
                        F[(var, lev)] = v
        for key, var, level in (("msl", "PRMSL", "mean sea level"), ("sst", "TMP", "surface"), ("land", "LAND", "surface")):
            if key not in F:
                v = _field(get, stem, idx, var, level)
                if v is not None:
                    F[key] = v
    return F


def _gc(la1, lo1, la2, lo2):
    p1, p2 = np.deg2rad(la1), np.deg2rad(la2)
    return 6371.0 * np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(np.deg2rad(lo2 - lo1)), -1, 1))


def pi_at(F, lat, lon, levels=None):
    from tcpyPI import pi as bepi
    lon = lon % 360
    i = np.where(np.abs(LAT - lat) <= 9)[0]; dl = ((LON - lon + 180) % 360) - 180
    j = np.where(np.abs(dl) <= 9 / max(np.cos(np.deg2rad(lat)), 0.2))[0]
    LA = LAT[i][:, None] + 0 * LON[j][None]; LO = LON[j][None] + 0 * LAT[i][:, None]
    W = np.cos(np.deg2rad(LA)); d = _gc(lat, lon, LA, LO); ring = (d >= 200) & (d <= 800)
    mean = lambda X, m: float(np.sum(X[m] * W[m]) / np.sum(W[m]))
    msl = mean(F["msl"][np.ix_(i, j)], ring) / 100.0
    sst = F["sst"][np.ix_(i, j)]; land = F["land"][np.ix_(i, j)] if "land" in F else np.zeros_like(sst)
    oc = (d <= 400) & (land < 0.5) & np.isfinite(sst) & (sst > 271.0)
    sstc = mean(sst, oc) - 273.15 if oc.sum() >= 2 else np.nan   # ERA5 test used >= 5 points at 0.25 deg; same area at 0.5 deg
    levs = [l for l in (levels or LEVELS) if ("TMP", l) in F and ("RH", l) in F and l <= min(1000, msl)]
    P = np.array(levs, float)
    TC = np.array([mean(F[("TMP", l)][np.ix_(i, j)], ring) for l in levs]) - 273.15
    RH = np.clip(np.array([mean(F[("RH", l)][np.ix_(i, j)], ring) for l in levs]), 0, 100)
    es = 6.112 * np.exp(17.67 * TC / (TC + 243.5)); e = RH / 100 * es; R = 621.97 * e / (P - e)   # mixing ratio, g/kg
    vmax = np.nan
    if np.isfinite(sstc):
        vmax = float(bepi(sstc, msl, P, TC, R, CKCD=0.9, ascent_flag=0, diss_flag=1, V_reduc=0.8, ptop=50, miss_handle=1)[0])
    return dict(pi_raw=vmax, pi=float(max(vmax, 10.0)) if np.isfinite(vmax) else 10.0, pi_floor=bool(not np.isfinite(vmax) or vmax < 10),
                sst_c=None if not np.isfinite(sstc) else round(sstc, 1), n_levels=len(levs))


def band(ratio):
    for hi, name, text in BANDS:
        if ratio < hi:
            return name, text


def write(day, outdir, get=None, base=None):
    import outlook
    get = get or outlook.get; base = base or outlook.CFG["sources"]["gefs"]
    ymd = f"{day:%Y%m%d}"; gp = os.path.join(outdir, f"outlook_{ymd}.geojson")
    storms = [f for f in json.load(open(gp))["features"] if f["properties"].get("kind") == "active_storm"] if os.path.exists(gp) else []
    out = dict(issued=f"{day:%Y-%m-%d}", note="BETA, display only. Wind / potential intensity (PI_RATIO_RESULT.md), GEFS port.", storms=[])
    if storms:
        F = fetch(get, base, ymd)
        for f in storms:
            p = f["properties"]; lo, la = f["geometry"]["coordinates"]; v = float(p["wind_kt"]) * 0.5144
            r = pi_at(F, la, lo); ratio = v / r["pi"]; name, text = band(ratio)
            out["storms"].append(dict(id=p.get("id"), name=p.get("name"), lat=la, lon=lo % 360, wind_ms=round(v, 1), pi_ms=round(r["pi"], 1),
                                      pi_floor=r["pi_floor"], sst_c=r["sst_c"], ratio=round(ratio, 2), band=name, text=text))
    tmp = os.path.join(outdir, f"ceiling_{ymd}.json.tmp")
    json.dump(out, open(tmp, "w"), separators=(",", ":")); os.replace(tmp, os.path.join(outdir, f"ceiling_{ymd}.json"))
    return out
