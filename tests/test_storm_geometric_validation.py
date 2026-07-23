"""
tests/test_storm_geometric_validation.py — unit tests for the geometric
validation harness. Synthetic data only; no ERA5.
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from storm_geometric_validation import (
    signed_loop_area,
    loop_features,
    evaluate,
    physics_guards,
    self_test,
)


# ------------------------------------------------------------------ #
# Test 1: signed_loop_area is correct and orientation-signed          #
# ------------------------------------------------------------------ #

def test_signed_loop_area_unit_square():
    """Counter-clockwise unit square has signed area +1; reversing the
    traversal flips the sign; fewer than 3 points gives 0."""
    x = [0.0, 1.0, 1.0, 0.0]
    y = [0.0, 0.0, 1.0, 1.0]
    ccw = signed_loop_area(x, y)
    cw = signed_loop_area(x[::-1], y[::-1])

    assert abs(ccw - 1.0) < 1e-9, f"CCW unit square area should be +1, got {ccw}"
    assert abs(cw + 1.0) < 1e-9, f"CW unit square area should be -1, got {cw}"
    assert signed_loop_area([0.0, 1.0], [0.0, 1.0]) == 0.0, "degenerate loop -> 0"


def test_area_norm_speed_invariance():
    """area_norm (area / path_len) should be (nearly) invariant to how densely
    the same geometric loop is sampled — the traversal-speed invariance."""
    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    x_coarse, y_coarse = np.cos(theta), np.sin(theta)
    theta_f = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    x_fine, y_fine = np.cos(theta_f), np.sin(theta_f)

    n_coarse = loop_features(x_coarse, y_coarse)["area_norm"]
    n_fine = loop_features(x_fine, y_fine)["area_norm"]
    # Both approximate the unit circle; area_norm should be close in magnitude.
    assert np.sign(n_coarse) == np.sign(n_fine)
    assert abs(abs(n_fine) - abs(n_coarse)) < 0.15, (
        f"area_norm should be roughly sampling-invariant "
        f"(coarse={n_coarse:.3f}, fine={n_fine:.3f})"
    )


# ------------------------------------------------------------------ #
# Test 2: pure noise -> inside null (verdict NOT supported)            #
# ------------------------------------------------------------------ #

def _verdict_supported(skill, guards):
    """Replicate run_on_data's gating logic without touching the sealed harness."""
    return (
        skill["pr_auc"]["p_value"] < 0.05
        and bool(guards["hemisphere_flip"]["passes"])
        and bool(guards["area_scaling"]["monotonic"])
    )


def test_noise_is_inside_null():
    """The reliable anti-overfitting signal on noise is that block-CV skill
    stays INSIDE the within-block permutation null, so the composite verdict
    is NOT supported. (Individual guards on noise are stochastic — the
    sign-flip guard is ~a coin flip when both correlations are ~0 — so we
    assert the deterministic composite verdict, not a single guard.)"""
    noise_res, _ = self_test(n=3000, seed=7, n_perm=40)
    skill = noise_res["skill"]
    guards = noise_res["guards"]

    assert skill["pr_auc"]["p_value"] >= 0.05, (
        f"pure noise must not beat the null (p={skill['pr_auc']['p_value']:.3f})"
    )
    assert not _verdict_supported(skill, guards), (
        "pure noise must not yield a SUPPORTED verdict"
    )


# ------------------------------------------------------------------ #
# Test 3: planted geometric signal -> above null AND guards pass       #
# ------------------------------------------------------------------ #

def test_planted_signal_is_above_null_and_guards_pass():
    _, sig_res = self_test(n=3000, seed=7, n_perm=40)
    skill = sig_res["skill"]
    guards = sig_res["guards"]

    assert skill["pr_auc"]["p_value"] < 0.05, (
        f"planted signal must beat the null (p={skill['pr_auc']['p_value']:.3f})"
    )
    assert guards["hemisphere_flip"]["passes"], (
        "hemisphere sign-flip guard should PASS on planted signal"
    )
    assert guards["area_scaling"]["monotonic"], (
        "area-scaling guard should be monotonic on planted signal"
    )
