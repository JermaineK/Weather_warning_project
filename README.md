# Weather Warning Project

Early-seed detection and intensity forecasting using **geometric-kernel** diagnostics (BORGT/GKA) on ERA5 and GPM IMERG.

**Goal:** detect organizing “seeds” *before* naming thresholds — reliably at **72 h** lead, with experimental horizons out to **10 days**.

---

## Features at a glance

- **Data ingestion:** ERA5 (winds, humidity, shear, SLP, SST proxy) and optional GPM IMERG (rainfall for confirmation).
- **Seed detection:** vorticity/divergence texture analysis with spatial size-laws.
- **Geometric invariants:** slope **2p**, knee **Aₖ**, and freedom-of-movement **F = (1 + VWS)⁻¹**.
- **Scoring:** a scalar **Seed Strength** \(S\) built from geometry + environment.
- **Forecast outputs:** per-lead alert grids, hindcast evaluation, auto-numbered reports (MD/TXT/PDF).

> Rainfall is treated as a **lagging confirmation** signal; predictors emphasise **pre-convective geometry** and **environment**.

---

## Install

```bash
git clone https://github.com/JermaineK/Weather_warning_project.git
cd Weather_warning_project

# Python env (Windows)
python -m venv .venv
.\.venv\Scripts\activate

# Install deps
pip install -r requirements.txt