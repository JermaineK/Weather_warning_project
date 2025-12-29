import math
import pandas as pd
import numpy as np

from alerts_logic_subprocess.throttle_by_percentile import _deterministic_keep_mask


def _grid(rows_per_lat: int = 4) -> pd.DataFrame:
    time = pd.Timestamp("2025-01-01 00:00:00")
    lats = np.repeat(np.arange(5, dtype=float), rows_per_lat)
    lons = np.tile(np.arange(rows_per_lat, dtype=float), 5)
    return pd.DataFrame({"time_h": time, "lat": lats, "lon": lons})


def test_keep_quantile_sampling_spreads_latitudes():
    elig = _grid(rows_per_lat=4)
    target = math.ceil(0.2 * len(elig))
    keep_counts = {elig["time_h"].iloc[0]: target}

    mask = _deterministic_keep_mask(
        elig=elig,
        keep_counts=keep_counts,
        keep_quantile=0.2,
        protected_idx=pd.Index([]),
    )
    kept = elig.loc[mask]

    assert kept["lat"].nunique() >= 3
    assert len(kept) == target


def test_keep_quantile_sampling_is_deterministic_to_order():
    base = _grid(rows_per_lat=2)
    keep_counts = {base["time_h"].iloc[0]: math.ceil(0.5 * len(base))}

    mask1 = _deterministic_keep_mask(
        elig=base,
        keep_counts=keep_counts,
        keep_quantile=0.5,
        protected_idx=pd.Index([]),
    )
    kept1 = set(zip(base.loc[mask1, "lat"], base.loc[mask1, "lon"]))

    shuffled = base.sample(frac=1.0, random_state=123).reset_index(drop=True)
    keep_counts2 = {shuffled["time_h"].iloc[0]: keep_counts[base["time_h"].iloc[0]]}

    mask2 = _deterministic_keep_mask(
        elig=shuffled,
        keep_counts=keep_counts2,
        keep_quantile=0.5,
        protected_idx=pd.Index([]),
    )
    kept2 = set(zip(shuffled.loc[mask2, "lat"], shuffled.loc[mask2, "lon"]))

    assert kept1 == kept2
