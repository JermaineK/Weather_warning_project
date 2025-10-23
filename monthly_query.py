#!/usr/bin/env python3
import cdsapi, calendar, os

dataset = "reanalysis-era5-single-levels"
variables = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_precipitation",
    "surface_latent_heat_flux",
    "surface_sensible_heat_flux",
    "total_column_water_vapour",
]

# Broaden westward so you cover 118–132E track points
AREA = [-8, 100, -45, 160]     # [North, West, South, East]
GRID = [0.25, 0.25]
YEAR = 2025
MONTHS = [4]             # Feb–Apr (add/remove as needed)
OUTDIR = "data_era5/monthly_nc"

os.makedirs(OUTDIR, exist_ok=True)
c = cdsapi.Client()

for m in MONTHS:
    ndays = calendar.monthrange(YEAR, m)[1]
    req = {
        "product_type": "reanalysis",
        "variable": variables,
        "year": str(YEAR),
        "month": f"{m:02d}",
        "day": [f"{d:02d}" for d in range(1, ndays+1)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "format": "netcdf",
        "area": AREA,
        "grid": GRID,
    }
    target = os.path.join(OUTDIR, f"era5_single_{YEAR}{m:02d}_au_wide_0p25.nc")
    print(f"[CDS] requesting {target}")
    c.retrieve(dataset, req, target=target)
    print(f"[CDS] wrote  {target}")