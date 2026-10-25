"""Run the outlook on your own computer, resuming after any shutdown from where it left off.

    python run_local.py --out C:/storm_outlook_data            # catch up to the latest forecast, verify, exit
    python run_local.py --out C:/storm_outlook_data --loop     # stay running; wake daily after the 00 UTC forecast
    python run_local.py --out C:/storm_outlook_data --start 2026-10-01   # first run: begin from a chosen date

How resuming works: state.json is written last for each day, and its date is the only record of progress.
On start the runner processes every day after that date, in order, up to the newest complete forecast.
A day interrupted mid-way is simply redone. NOAA keeps past GEFS forecasts online, so missed days can be filled in.
Licence: PolyForm Noncommercial 1.0.0.
"""
import argparse, json, os, sys, time, traceback
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import outlook  # noqa: E402

READY_KEY = "pgrb2bp5/gec00.t00z.pgrb2b.0p50.f120.idx"   # last file the outlook needs from each 00 UTC run


def forecast_ready(day):
    url = f"{outlook.CFG['sources']['gefs']}/gefs.{day:%Y%m%d}/00/atmos/{READY_KEY}"
    try:
        return outlook.SES.head(url, timeout=30).status_code == 200
    except Exception:
        return False


def last_done(out):
    sp = os.path.join(out, "state.json")
    if not os.path.exists(sp):
        return None
    return datetime.strptime(json.load(open(sp))["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)


def log(out, msg):
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC  {msg}"
    print(line, flush=True)
    with open(os.path.join(out, "run_log.txt"), "a") as f:
        f.write(line + "\n")


def catch_up(out, start=None, retries=3):
    os.makedirs(out, exist_ok=True)
    import shutil                                    # always refresh the map page, so app updates reach it
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "site", "index.html"), os.path.join(out, "index.html"))
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    done = last_done(out)
    day = done + timedelta(days=1) if done else (start or today - timedelta(days=4))
    n = 0
    while day <= today:
        if not forecast_ready(day):
            if day >= today - timedelta(days=1):
                log(out, f"{day:%Y-%m-%d}: forecast not complete yet; will resume here next time")
                break
            log(out, f"{day:%Y-%m-%d}: forecast missing on NOAA's server; skipped (region ages restart)")
            day += timedelta(days=1)
            continue
        for k in range(retries):
            try:
                outlook.run(day, out)
                n += 1
                break
            except Exception as e:
                log(out, f"{day:%Y-%m-%d}: attempt {k + 1} failed: {type(e).__name__}: {str(e)[:120]}")
                time.sleep(30 * (k + 1))
        else:
            log(out, f"{day:%Y-%m-%d}: giving up for now; will retry from here next time")
            break
        day += timedelta(days=1)
    try:
        import verify
        verify.score(out)
    except Exception as e:
        log(out, f"verification skipped: {type(e).__name__}: {str(e)[:120]}")
    log(out, f"processed {n} day(s); last completed {last_done(out):%Y-%m-%d}" if last_done(out) else f"processed {n} day(s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outlook_local")
    ap.add_argument("--start", help="first issue date on a fresh output folder (default: 4 days ago)")
    ap.add_argument("--loop", action="store_true", help="keep running and update daily")
    ap.add_argument("--members", type=int, default=10, help="ensemble members for the agreement layer (0 = off)")
    a = ap.parse_args()
    start = datetime.strptime(a.start, "%Y-%m-%d").replace(tzinfo=timezone.utc) if a.start else None
    while True:
        try:
            catch_up(a.out, start)
            try:                                      # dashboard: forecast-hour and ensemble views (display only)
                import forecast_view
                forecast_view.make(a.out, members=a.members, ensemble=a.members > 0)
            except Exception as e:
                log(a.out, f"forecast views skipped ({type(e).__name__}: {e})")
            try:                                      # animated GIF of every day in the folder
                import make_gif
                make_gif.make(a.out)
            except Exception as e:
                log(a.out, f"GIF skipped ({type(e).__name__}: {e})")
        except Exception:
            log(a.out, "unexpected error:\n" + traceback.format_exc())
        if not a.loop:
            return
        now = datetime.now(timezone.utc)
        nxt = now.replace(hour=6, minute=40, second=0, microsecond=0)
        if nxt <= now:
            nxt += timedelta(days=1)
        if last_done(a.out) and last_done(a.out) < now.replace(hour=0, minute=0, second=0, microsecond=0):
            nxt = min(nxt, now + timedelta(minutes=30))  # still behind: retry soon
        log(a.out, f"sleeping until {nxt:%Y-%m-%d %H:%M} UTC")
        time.sleep(max(60, (nxt - now).total_seconds()))


if __name__ == "__main__":
    main()
