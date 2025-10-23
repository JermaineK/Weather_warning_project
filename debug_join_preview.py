# debug_join_preview.py
import pandas as pd
from pathlib import Path

f_csv = "data/features_era5_au.csv"
l_csv = "data/storm_labels_2025-04.csv"

df = pd.read_csv(f_csv)
lb = pd.read_csv(l_csv)

# parse times
df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
lb["time"] = pd.to_datetime(lb["time"], utc=True, errors="coerce")

print("=== FEATURES ===")
print("rows:", len(df))
print("time.min:", df["time"].min(), "time.max:", df["time"].max())
print("lat_c range:", df["lat_c"].min(), "→", df["lat_c"].max() if "lat_c" in df.columns else "MISSING")
print("lon_c range:", df["lon_c"].min(), "→", df["lon_c"].max() if "lon_c" in df.columns else "MISSING")
if "lat" in df.columns and "lon" in df.columns:
    print("lat range:", df["lat"].min(), "→", df["lat"].max())
    print("lon range:", df["lon"].min(), "→", df["lon"].max())
print("\n=== LABELS ===")
print("rows:", len(lb))
print("time.min:", lb["time"].min(), "time.max:", lb["time"].max())
print("lat range:", lb["lat"].min(), "→", lb["lat"].max())
print("lon range:", lb["lon"].min(), "→", lb["lon"].max())

print("\nSAMPLE labels (first 5):")
print(lb.head(5).to_string(index=False))