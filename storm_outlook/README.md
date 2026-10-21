# Storm formation outlook

A free, lightweight daily outlook of **where new tropical cyclones may form in the next 5 days**, over tropical and subtropical oceans worldwide (10–35° latitude). It runs once a day from public NOAA forecast data, needs no server, and can be hosted for free on GitHub.

> **Experimental research product — not an official warning.** Always follow your national meteorological service (e.g. Bureau of Meteorology, NHC, JTWC, JMA, IMD, Météo-France, Fiji Meteorological Service). This outlook does not forecast storm strength, track or impacts.

Free for non-commercial use under the [PolyForm Noncommercial License 1.0.0](LICENSE.md). It must never be sold or charged for.

## What it shows
- **Alert regions** where formation is possible, tracked from day to day. Only regions of at least 6 grid points that persist at least 3 days are shown.

  | level | rule | in the 2017–19 test, regions like this became storms |
  |---|---|---|
  | yellow | passes the cluster rule | about 4 in 10 |
  | orange | ≥ 12 points and ≥ 5 days | about 5 in 10 |
  | red | ≥ 25 points, ≥ 8 days, peak chance ≥ 35% | about 8 in 10 |

- **Chance of formation** within 5° in the next 3 days (shaded dots).
- **Active storms**, from NCEP's TC vitals file (NHC and JTWC).

Overall, in 2017–19 forecasts the outlook alerted **84.9% of storms at least a day ahead** (just short of the 85% target set before the alert level was chosen; no tested level met it), with about 330 false-alarm regions per January–September season worldwide. Official outlooks are considerably more accurate. The first fully independent check is October–December 2026.

## How it works
1. Downloads 35 fields from the **NOAA GEFS v12 control forecast** (00 UTC; 3, 24, 48, 96, 120 h) by HTTP byte range from NOAA's free public bucket: about 6 MB and 25 s per day.
2. Computes 33 features at 2,301 ocean points:
   - low-level spin;
   - deep wind shear;
   - area of strong instability within reach;
   - moisture;
   - latitude and season.
3. **Instability gate:** no alerts unless CAPE > 1000 J/kg lies within 12°.
4. **Model:** boosted decision trees (78 trees for days 0–3, 45 for days 3–5), trained on the 2010–16 GEFS reforecast and calibrated. The trees are stored as plain arrays (`model/trees.npz`, 93 kB) and evaluated with numpy, so no machine-learning library is needed.
5. **Existing-storm corrections:**
   - no new-storm alert within 10° of an active system;
   - probabilities on the map lowered 5–10° from active systems;
   - no alerts 800–2,500 km west-poleward of storms ≥ 50 kt.
6. **Cluster rule and levels** as above.

Every constant is in `config.json`. The method and its tests are documented in the parent project.

## Does it learn from confirmed storms?
Yes, through checking rather than silently retuning itself.
- **Every day (automatic):** `verify.py` matches past outlooks with storms confirmed in IBTrACS. It scores hits, misses and false alarms, writes `verification.json` and `verification_storms.csv`, and shows a live scorecard on the map.
  - It raises a **drift flag** if performance leaves the range seen in 2017–19, for example if a NOAA data change breaks something.
  - On June–September 2026 it reproduced the research scoring exactly: 48 storms, 85% alerted at least 24 h ahead.
- **Every day (automatic):** the inputs are archived (`archive/features_*.npz`, about 240 kB a day, about 90 MB a year) so the model can be re-examined later.
- **Once a year (human-reviewed):** `refit_review.py` fits a candidate recalibration on the archive and scores it on the most recent months. It writes `refit_proposal.json` and never edits the model. A change should be adopted only by pull request, after a pre-registered blind check.

**Why the model does not retune itself:** I tested monthly self-recalibration from all confirmed outcomes, in a simulated deployment over 2017–19 and live 2026. In both it was slightly *worse* than the frozen model (log-loss +0.1% and +0.6%). The probability errors are S-shaped (too high at both ends, too low in the middle), and a small automatic adjustment can't fix both ends at once. Storms are also rare, about 80–90 a year worldwide, so monthly retraining would mostly chase noise. Learning only from "hits" would be worse still: it would push probabilities up everywhere, because misses and false alarms are what keep them honest.

## Host your own copy (free)
1. **Fork** this repository, or copy the `storm_outlook/` folder and `.github/workflows/storm_outlook.yml` into a new repository.
2. In **Settings → Actions → General**, allow workflows, and give them read and write permission.
3. Run the workflow once from the **Actions** tab ("storm-outlook" → "Run workflow"). Scheduled runs only start once the workflow is on the default branch.
4. In **Settings → Pages**, publish from the branch `outlook-site`, folder `/ (root)`.
5. Your map is at `https://<user>.github.io/<repository>/`. It updates daily at about 06:40 UTC.

The first run processes the 4 previous days, so regions have their history. The page uses OpenStreetMap tiles and Leaflet from a public CDN; for heavy public traffic, use a tile provider that permits it.

## Run it locally
```
conda install -c conda-forge python-eccodes numpy scipy requests      # or: apt install libeccodes-dev && pip install -r requirements.txt
python outlook.py --out site            # today's 00 UTC outlook (after ~06 UTC)
python outlook.py --date 2026-10-15 --out site
```
Then open `site/index.html` through a local web server (`python -m http.server -d site`).

## Data sources and credits
- NOAA Global Ensemble Forecast System (GEFS v12), via the NOAA Open Data Dissemination programme.
- NCEP tropical cyclone vitals (from NHC and JTWC); fallback: IBTrACS v04r01 ACTIVE (NOAA NCEI), which lags 2–3 days.
- Training and verification storms: IBTrACS v04r01.

## Known limits
- **Coverage:** no coverage between 10°S and 10°N, or beyond 35° latitude.
- **Missing storm list:** if TC vitals is missing on a day with active storms, the lagged fallback is used and the masks may be wrong. The page shows which source was used.
- **Probabilities:** the highest probabilities are over-confident (they verify about half the time), so levels are shown rather than raw percentages.
- **Fixed model:** the model is frozen. It should be re-verified each season against the storm record before any change.
