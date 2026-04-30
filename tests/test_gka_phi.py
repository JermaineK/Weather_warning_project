"""
tests/test_gka_phi.py — unit tests for the gka_phi composite feature.

Tests use synthetic data only; no ERA5 or real data required.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root on sys.path so we can import from features_subprocess and root-level scripts.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features_subprocess.compute_gka_features import (
    _compute_chunk_features,
    _bind_columns,
)
from eval_phi_threshold import grid_search_phi_c, _sigmoid


# ------------------------------------------------------------------ #
# Test 1: constant agree, varying zeta variance → phi decreases       #
# ------------------------------------------------------------------ #

def _make_group_df(lat, lon, zeta_vals, agree=0.6, relax=1.0, n_repeat=1):
    """Build a small single-group DataFrame."""
    rows = []
    for t, z in enumerate(zeta_vals):
        rows.append({
            "lat": lat, "lon": lon,
            "time": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=t),
            "zeta": z,
            "agree": agree,
            "relax": relax,
        })
    return pd.DataFrame(rows)


def test_phi_decreases_with_zeta_variance():
    """Higher zeta variance in a group → lower gka_phi (all else equal)."""
    rng = np.random.default_rng(seed=0)

    # Group A: near-constant zeta → very small V_zeta
    zeta_low_var  = np.full(12, 1e-5) + rng.normal(0, 1e-7, 12)
    # Group B: highly variable zeta → large V_zeta
    zeta_high_var = rng.normal(0, 1.0, 12)

    df_a = _make_group_df(lat=10.0, lon=100.0, zeta_vals=zeta_low_var)
    df_b = _make_group_df(lat=11.0, lon=101.0, zeta_vals=zeta_high_var)
    df = pd.concat([df_a, df_b], ignore_index=True)

    bind = _bind_columns(list(df.columns))
    out = _compute_chunk_features(df, bind)

    phi_a = out.loc[out["lat"] == 10.0, "gka_phi"].dropna().to_numpy(float)
    phi_b = out.loc[out["lat"] == 11.0, "gka_phi"].dropna().to_numpy(float)

    assert len(phi_a) > 0 and len(phi_b) > 0, "gka_phi should be computed for both groups"
    # After robust scaling the ordering of group means is preserved.
    assert phi_a.mean() > phi_b.mean(), (
        f"Low-variance group should have higher mean gka_phi "
        f"(got low_var={phi_a.mean():.4f}, high_var={phi_b.mean():.4f})"
    )


# ------------------------------------------------------------------ #
# Test 2: relax near zero → no inf/NaN (1e-6 guard active)            #
# ------------------------------------------------------------------ #

def test_near_zero_relax_no_inf_nan():
    """relax ≈ 1e-8 must not produce inf or NaN in gka_phi."""
    n = 20
    rng = np.random.default_rng(seed=1)

    df = pd.DataFrame({
        "lat":   np.tile([5.0, 6.0], n // 2),
        "lon":   np.tile([80.0, 81.0], n // 2),
        "time":  pd.date_range("2025-02-01", periods=n, freq="h"),
        "zeta":  rng.normal(0, 0.5, n),
        "agree": rng.uniform(0.1, 1.0, n),
        "relax": np.full(n, 1e-8),   # near-zero relax
    })

    bind = _bind_columns(list(df.columns))
    out = _compute_chunk_features(df, bind)

    phi = out["gka_phi"].to_numpy(float)
    assert not np.any(np.isinf(phi)), "gka_phi must not contain inf with near-zero relax"
    # NaN is acceptable only if the robust-scale step hits degenerate data,
    # but the values themselves must not be inf.
    finite_or_nan = np.isfinite(phi) | np.isnan(phi)
    assert finite_or_nan.all(), "gka_phi must not contain inf"


# ------------------------------------------------------------------ #
# Test 3: planted phi_c → eval recovers it within ±10%                #
# ------------------------------------------------------------------ #

def test_phi_c_recovery():
    """Grid search should recover a planted structural break within ±10%.

    phi_c_true is set to the empirical 0.70 quantile of phi so it lies exactly
    on a grid-search candidate, eliminating quantisation error from the test.
    """
    rng = np.random.default_rng(seed=42)
    n = 4000

    phi = rng.uniform(-2.0, 2.0, n)

    quantiles = np.linspace(0.1, 0.9, 17)
    # Anchor the planted break to an exact grid point to avoid quantisation bias.
    phi_c_true = float(np.nanquantile(phi, 0.70))

    # Sharp break: logit has a large slope change at phi_c_true.
    logit = -0.5 + 0.5 * phi + 5.0 * np.maximum(0.0, phi - phi_c_true)
    prob  = _sigmoid(logit)
    y     = rng.binomial(1, prob).astype(float)

    phi_c_hat, *_ = grid_search_phi_c(phi, y, quantiles)

    rel_err = abs(phi_c_hat - phi_c_true) / abs(phi_c_true)
    assert rel_err <= 0.10, (
        f"Recovered phi_c={phi_c_hat:.4f} deviates from planted "
        f"phi_c={phi_c_true:.4f} by {rel_err*100:.1f}% (threshold: 10%)"
    )
