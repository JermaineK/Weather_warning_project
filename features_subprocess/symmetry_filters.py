"""
symmetry_filters.py

Agent: repurpose OAM/GKA symmetry kernels for weather-grid patches without changing math.

Small, pure kernels for parity-odd contrast, spiral harmonics, and log-spiral
projections. These are linear in the data; keep them stable and testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.signal import savgol_filter


def parity_odd_contrast_ring(I_r_theta: np.ndarray, eps: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parity-odd contrast per radius for a polar grid (r, theta).

    eta_odd(r) = |mean_theta 0.5*(I - I(theta+pi))| / (mean_theta 0.5*(I + I(theta+pi)) + eps)
    Returns (eta_odd, mean_even).
    """
    if I_r_theta.ndim != 2:
        raise ValueError("I_r_theta must be 2D (n_r, n_theta)")
    n_theta = I_r_theta.shape[1]
    I_roll = np.roll(I_r_theta, shift=n_theta // 2, axis=1)
    I_even = 0.5 * (I_r_theta + I_roll)
    I_odd = 0.5 * (I_r_theta - I_roll)
    mean_even = I_even.mean(axis=1)
    mean_odd = I_odd.mean(axis=1)
    eta_odd = np.abs(mean_odd) / (mean_even + eps)
    return eta_odd, mean_even


def spiral_harmonic_contrast(I_r_theta: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Spiral harmonic contrast using m=1,2 angular Fourier modes (imag parts).

    eta_spiral(r) = (Im a1)^2 + (Im a2)^2 / (total power + eps).
    """
    if I_r_theta.ndim != 2:
        raise ValueError("I_r_theta must be 2D (n_r, n_theta)")
    n_theta = I_r_theta.shape[1]
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    basis1 = np.exp(-1j * theta)
    basis2 = np.exp(-2j * theta)
    a1 = (I_r_theta * basis1).mean(axis=1)
    a2 = (I_r_theta * basis2).mean(axis=1)
    p_odd = (a1.imag**2 + a2.imag**2)
    p_tot = (np.abs(I_r_theta) ** 2).mean(axis=1)
    return p_odd / (p_tot + eps)


def log_spiral_kernel(r_vals: np.ndarray, theta_vals: np.ndarray, alpha: float) -> np.ndarray:
    """Fixed log-spiral kernel K_alpha(r,theta) = cos(alpha ln r - theta)."""
    R, T = np.meshgrid(r_vals, theta_vals, indexing="ij")
    return np.cos(alpha * np.log(R + 1e-9) - T)


def kernel_projection(I_r_theta: np.ndarray, r_vals: np.ndarray, alpha: float, eps: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project onto a fixed log-spiral kernel with pitch alpha.

    Returns (eta_kernel, mean_even) where eta_kernel = |<I*K>| / (mean_even + eps).
    """
    n_theta = I_r_theta.shape[1]
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    K = log_spiral_kernel(r_vals, theta, alpha)
    proj = (I_r_theta * K).mean(axis=1)
    eta_odd, mean_even = parity_odd_contrast_ring(I_r_theta, eps)
    eta_kernel = np.abs(proj) / (mean_even + eps)
    return eta_kernel, mean_even


def slow_tick_projection(series: np.ndarray, dt: float, f_star: float) -> float:
    """Projection magnitude of a time series onto a sinusoid at f_star."""
    series = np.asarray(series, float)
    n = len(series)
    t = np.arange(n) * dt
    omega = 2 * np.pi * f_star
    A = (2.0 / n) * np.sum(series * np.cos(omega * t))
    B = (2.0 / n) * np.sum(series * np.sin(omega * t))
    return float(np.sqrt(A**2 + B**2))


@dataclass
class KneeResult:
    r_knee: float
    d2: np.ndarray
    log_r: np.ndarray
    log_eta: np.ndarray


def knee_on_loglog(r: np.ndarray, eta: np.ndarray, window: int = 9, polyorder: int = 2) -> KneeResult:
    """
    Knee via curvature on log-log: smooth log(eta) vs log(r) with Savitzky–Golay, then find max |d2|.
    """
    mask = (eta > 0) & (r > 0) & np.isfinite(eta) & np.isfinite(r)
    if mask.sum() < 3:
        return KneeResult(r_knee=np.nan, d2=np.array([]), log_r=np.array([]), log_eta=np.array([]))
    log_r = np.log(r[mask])
    log_eta = np.log(eta[mask])
    if log_r.size < window:
        window = max(5, 2 * (log_r.size // 2) + 1)
    d1 = np.gradient(log_eta, log_r)
    d1s = savgol_filter(d1, window_length=max(3, window), polyorder=polyorder)
    d2 = np.gradient(d1s, log_r)
    idx = int(np.argmax(np.abs(d2)))
    return KneeResult(r_knee=float(np.exp(log_r[idx])), d2=d2, log_r=log_r, log_eta=log_eta)
