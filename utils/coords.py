import numpy as np
import pandas as pd

def utc_naive(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None)

def norm_lon(series: pd.Series, mode: str="-180..180") -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if mode == "none": return x
    if mode == "0..360":
        y = (x % 360 + 360) % 360
        return y
    # default -180..180
    return ((x + 180) % 360) - 180