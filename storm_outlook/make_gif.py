"""Animated GIF of every outlook day in an output folder (alerts, hits/misses, storm tracks, explosive lows).

    python make_gif.py --out C:/storm_outlook_data            # writes outlook_animation.gif in that folder
    python make_gif.py --out C:/storm_outlook_data --days 30   # only the last 30 days

run_local.py calls this automatically after each update. Needs matplotlib and pillow.
Licence: PolyForm Noncommercial 1.0.0.
"""
import argparse, json, os
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COL = {"yellow": "#f2c400", "orange": "#ef7d00", "red": "#d62728"}
GREEN, GREY, RED, PUR = "#2e7d32", "#8a8a8a", "#c62828", "#6a1b9a"


def _dnum(d):
    return datetime.strptime(d, "%Y%m%d").toordinal()


def _cells(f):
    return np.array([np.array(c[0])[:4].mean(0) for c in f["geometry"]["coordinates"]])


def _dlon(a, b):
    return (a - b + 180) % 360 - 180


def make(outdir, max_days=None, fname="outlook_animation.gif", ms_per_frame=1100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon
    from matplotlib.collections import PatchCollection
    from matplotlib.lines import Line2D
    from PIL import Image

    D = sorted(f[8:16] for f in os.listdir(outdir) if f.startswith("outlook_") and f.endswith(".geojson") and f[8:16].isdigit())
    if max_days:
        D = D[-max_days:]
    if not D:
        print("no outlook files found"); return None
    G = {d: json.load(open(os.path.join(outdir, f"outlook_{d}.geojson"))) for d in D}
    load = lambda n: json.load(open(os.path.join(outdir, n))) if os.path.exists(os.path.join(outdir, n)) else None
    MV, BV = load("map_verification.json"), load("bomb_verification.json")
    BB = {d: load(f"bomb_{d}.json") for d in D}
    land = json.load(open(os.path.join(HERE, "model", "land.geojson")))
    landp = [np.array(pg[0]) for f in land["features"]
             for pg in (f["geometry"]["coordinates"] if f["geometry"]["type"] == "MultiPolygon" else [f["geometry"]["coordinates"]])]
    alerts = lambda d: [f for f in G[d]["features"] if f["properties"]["kind"] == "alert"]
    tracks = {}
    frames = []
    for d in D:
        for f in G[d]["features"]:
            if f["properties"]["kind"] == "active_storm":
                x, y = f["geometry"]["coordinates"]; tracks.setdefault(f["properties"]["id"], []).append((d, (x + 180) % 360 - 180, y, f["properties"]))
    for d in D:
        fig, ax = plt.subplots(figsize=(11, 4.6), dpi=100)
        ax.set_facecolor("#cfe3ef")
        ax.add_collection(PatchCollection([Polygon(p, closed=True) for p in landp], fc="#f1eee6", ec="#b9b4a8", lw=0.4, zorder=1))
        ax.axhspan(-10, 10, color="#9aa", alpha=0.12, zorder=1.5, lw=0)
        # explosive-low chance (purple)
        if BB.get(d):
            A = np.array(BB[d]["alerts"]).reshape(-1, 3)
            if len(A):
                ax.add_collection(PatchCollection([Rectangle(((lo + 180) % 360 - 180 - 1.25, la - 1.25), 2.5, 2.5) for la, lo, _ in A],
                                                  fc=PUR, ec="none", alpha=0.35, zorder=2))
        # tropical alert regions, green once their storm has formed (as known on this day)
        stat = {}
        if MV:
            for r in MV["regions"]:
                if r["day"] == d:
                    became = [b for b in r["became"] if b["day"] <= d]
                    stat[r["index"]] = "hit" if became else None
        for i, f in enumerate(alerts(d)):
            c = GREEN if stat.get(i) == "hit" else COL[f["properties"]["level"]]
            ax.add_collection(PatchCollection([Rectangle((x - 1.25, y - 1.25), 2.5, 2.5) for x, y in _cells(f)], fc=c, ec=c, alpha=0.55, lw=0.3, zorder=3))
        # past alerts that came true (green ghost, up to 5 days) and expired alerts (grey, on the day they expire)
        if MV:
            cur = [_cells(f).mean(0) for f in alerts(d)]
            for r in MV["regions"]:
                if r["day"] not in G or r["day"] >= d or _dnum(d) - _dnum(r["day"]) > 5:
                    continue
                f = alerts(r["day"])[r["index"]] if r["index"] < len(alerts(r["day"])) else None
                if f is None:
                    continue
                cc = _cells(f)
                hit = any(b["day"] <= d for b in r["became"])
                if hit and not any(np.hypot(_dlon(q[0], cc.mean(0)[0]), q[1] - cc.mean(0)[1]) <= 8 for q in cur):
                    ax.add_collection(PatchCollection([Rectangle((x - 1.25, y - 1.25), 2.5, 2.5) for x, y in cc], fc=GREEN, ec=GREEN, alpha=0.3, lw=0.3, zorder=2.6))
                elif r["status"] == "false_alarm" and _dnum(d) - _dnum(r["day"]) == 5 and not any(np.hypot(_dlon(q[0], cc.mean(0)[0]), q[1] - cc.mean(0)[1]) <= 8 for q in cur):
                    ax.add_collection(PatchCollection([Rectangle((x - 1.25, y - 1.25), 2.5, 2.5) for x, y in cc], fc=GREY, ec=GREY, alpha=0.25, lw=0.5, zorder=2.5))
        # storms: track so far, position, 24-h motion
        for sid, pts in tracks.items():
            P = [(x, y, pr) for dd, x, y, pr in pts if dd <= d]
            if not any(dd == d for dd, *_ in pts):          # only storms in today's storm list
                continue
            xs, ys = np.array([p[0] for p in P]), np.array([p[1] for p in P])
            for seg in np.split(np.c_[xs, ys], np.where(np.abs(np.diff(xs)) > 180)[0] + 1):
                if len(seg) > 1: ax.plot(seg[:, 0], seg[:, 1], color="#222", lw=1.0, zorder=5)
            x, y, pr = P[-1]
            ax.plot(x, y, "o", ms=6, mfc="white", mec="#222", mew=1.3, zorder=6)
            if pr.get("motion_deg") is not None and pr.get("motion_ms"):
                th = np.deg2rad(pr["motion_deg"]); km = pr["motion_ms"] * 86.4
                dy = km * np.cos(th) / 111.2; dx = km * np.sin(th) / (111.2 * max(np.cos(np.deg2rad(y)), 0.2))
                ax.plot([x, x + dx], [y, y + dy], ":", color="#222", lw=1.1, zorder=5)
            ax.text(x + 1.5, y + 1.5, f"{pr['name']} {pr['wind_kt']} kt", fontsize=6, zorder=7, color="#111")
        # labels: formations (last 7 days) and explosive lows (last 3 days)
        labs = []
        if MV:
            for fm in MV["formations"]:
                if fm["day"] <= d and _dnum(d) - _dnum(fm["day"]) <= 7:
                    ok = fm.get("hit")
                    labs.append((fm["lat"], (fm["lon"] + 180) % 360 - 180, GREEN if ok else RED,
                                 f"✓ {fm['name']}: alerted {fm.get('lead_days', 0):.0f} d ahead" if ok else f"✗ {fm['name']}: no alert"))
        if BV:
            for e in BV["events"]:
                if e.get("scorable") and not e.get("tropical_cyclone") and e["start"] <= d and _dnum(d) - _dnum(e["start"]) <= 3:
                    ok = e["alerted_days_before"] > 0
                    labs.append((e["lat"], (e["lon"] + 180) % 360 - 180, PUR if ok else RED,
                                 f"{'✓' if ok else '✗'} explosive low −{e['fall_hpa']:.0f} hPa"))
        placed = []
        for la, lo, c, txt in labs:
            x, y = lo + 2, la + 2
            while any(abs(_dlon(x, a)) < 35 and abs(y - b) < 5 for a, b in placed):
                y -= 6
            placed.append((x, y))
            right = lo > 110                                  # near the right edge: put the label to the left
            ax.annotate(txt, (lo, la), (lo - 2 if right else x, y), ha="right" if right else "left", fontsize=6.3, color="white", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.25", fc=c, ec="white", lw=0.8),
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.8))
        ax.set_xlim(-180, 180); ax.set_ylim(-72, 72); ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-60, 61, 30))
        ax.tick_params(labelsize=6.5)
        ax.set_title(f"Storm formation outlook, issued {d[:4]}-{d[4:6]}-{d[6:]} 00 UTC  (experimental; follow official warnings)", loc="left", fontsize=9)
        hh = [Rectangle((0, 0), 1, 1, fc=COL[k], alpha=0.6) for k in ("yellow", "orange", "red")] + \
             [Rectangle((0, 0), 1, 1, fc=GREEN, alpha=0.6), Rectangle((0, 0), 1, 1, fc=GREY, alpha=0.3), Rectangle((0, 0), 1, 1, fc=PUR, alpha=0.4),
              Line2D([], [], marker="o", color="#222", mfc="white")]
        ax.legend(hh, ["yellow alert", "orange alert", "red alert", "alert came true", "alert expired, no storm",
                       "explosive-low chance (3 days)", "storm: track, 24 h motion"],
                  loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4, fontsize=6.5, frameon=False)
        ax.text(178, -70, "grey band 10°S–10°N: outside the tropical model's grid", ha="right", va="bottom", fontsize=5.8, color="#566")
        fig.tight_layout(); fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]).quantize(colors=128, method=Image.Quantize.MEDIANCUT))
        plt.close(fig)
    path = os.path.join(outdir, fname); tmp = path + ".tmp.gif"
    frames[0].save(tmp, save_all=True, append_images=frames[1:] + [frames[-1]] * 2, duration=ms_per_frame, loop=0, optimize=True)
    for k in range(5):                                # Windows: a viewer may hold the old file open briefly
        try:
            os.replace(tmp, path); break
        except PermissionError:
            import time; time.sleep(2)
    else:
        path = os.path.join(outdir, f"outlook_animation_{D[-1]}.gif"); os.replace(tmp, path)
    print(f"{path}: {len(D)} days, {os.path.getsize(path) // 1000} kB")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outlook_local")
    ap.add_argument("--days", type=int, help="only the last N days (default: all)")
    a = ap.parse_args()
    make(a.out, a.days)
