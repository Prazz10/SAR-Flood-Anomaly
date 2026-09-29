"""
Statistical feature construction for pixel-wise SAR time series, following
Section 3.2 of Ta et al. (2026), "SAR Flood Anomaly Mapping Through
Statistical Time-Series Feature Classification and Pixel-Wise TCEV Modeling",
Remote Sens. 2026, 18, 3323.

Pipeline per pixel (paper eqs. 1-6):
  1. sigma_dB = 10*log10(sigma0)                      (eq. 1, done upstream)
  2. sigma_dB' = -sigma_dB                            (eq. 2, sign inversion)
  3. sort ascending, empirical CDF F_i = i/(n+1)       (eq. 3-4)
  4. Gumbel reduced-variate transform:
        y_i = -ln(-ln(F_i))                            (eq. 5)
  5. piecewise-linear fit of y (vertical) vs sigma_dB' (horizontal), i.e. a
     two-segment Gumbel probability plot -> breakpoint + two line fits
  6. nine descriptors C1-C9 (eq. 6, prose in Sec 3.2)

NOTE on ambiguous descriptors: the paper describes C1-C9 in prose without
giving closed-form definitions for every one. Where the text is fully
explicit (C1-C7, from the two fitted line segments and their breakpoint) the
implementation below follows it directly. C8 (temporal variation amplitude)
and C9 (mean of upper 10% SAR responses) are reconstructed from the prose
description; see comments at each definition.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PiecewiseFit:
    """Result of the two-segment Gumbel-plot fit for one pixel's history."""
    x: np.ndarray          # sorted transformed backscatter, sigma_dB'
    y: np.ndarray          # Gumbel reduced variate, -ln(-ln F)
    breakpoint_idx: int    # index in x/y where the fit switches segments
    slope1: float
    intercept1: float
    slope2: float
    intercept2: float
    sse: float


def empirical_cdf_sorted(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort x ascending and compute the rank-based empirical CDF (paper eq. 3-4):
        F(x_(i)) = i / (n+1)
    Returns (x_sorted, F).
    """
    x_sorted = np.sort(x)
    n = len(x_sorted)
    ranks = np.arange(1, n + 1)
    F = ranks / (n + 1)
    return x_sorted, F


def gumbel_transform(F: np.ndarray) -> np.ndarray:
    """Double-log extreme-value transform (paper eq. 5): y = -ln(-ln F)."""
    return -np.log(-np.log(F))


def _linfit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """OLS line fit y = slope*x + intercept. Returns (slope, intercept, sse)."""
    if len(x) < 2:
        return 0.0, float(y[0]) if len(y) else 0.0, 0.0
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), residuals, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = slope * x + intercept
    sse = float(np.sum((y - pred) ** 2))
    return float(slope), float(intercept), sse


def fit_piecewise_linear(x: np.ndarray, y: np.ndarray, min_segment: int = 5) -> PiecewiseFit:
    """
    Fit a two-segment linear model to (x, y), searching over all valid
    breakpoints and keeping the split that minimizes total SSE. x and y must
    already be sorted ascending by x (guaranteed since x = sorted sigma_dB'
    and y is a monotonic transform of rank).

    This implements the "piecewise linear fitting strategy... to identify
    the transition point between regular variations and extreme behaviors"
    (Sec 3.2).
    """
    n = len(x)
    if n < 2 * min_segment:
        # too few points to split meaningfully -- fit a single line, and just
        # duplicate it into "segment 2" so C1-C7 are still well-defined
        slope, intercept, sse = _linfit(x, y)
        return PiecewiseFit(x, y, n - 1, slope, intercept, slope, intercept, sse)

    best = None
    for bp in range(min_segment, n - min_segment + 1):
        x1, y1 = x[:bp], y[:bp]
        x2, y2 = x[bp - 1:], y[bp - 1:]  # share the breakpoint sample so segments meet
        s1, i1, sse1 = _linfit(x1, y1)
        s2, i2, sse2 = _linfit(x2, y2)
        total_sse = sse1 + sse2
        if best is None or total_sse < best.sse:
            best = PiecewiseFit(x, y, bp - 1, s1, i1, s2, i2, total_sse)
    return best


def compute_descriptors(fit: PiecewiseFit, sigma_dB_prime_full: np.ndarray) -> np.ndarray:
    """
    Compute the nine statistical descriptors C1-C9 from a piecewise fit and
    the full transformed (sign-inverted) historical series.

    C1: slope ratio between the two segments (extreme / regular)
    C2, C3: slope, intercept of the first (regular) segment
    C4, C5: slope, intercept of the second (extreme) segment
    C6, C7: breakpoint descriptors -- the (x, y) coordinates of the
            transition point, i.e. the sigma_dB' value and reduced variate
            at which the series shifts from regular to extreme behavior
    C8: temporal variation amplitude = max - min of the transformed series
        (reconstructed from "quantifying the magnitude of SAR backscatter
        changes over the observation period")
    C9: mean of the upper 10% of the transformed series (reconstructed from
        "mean value of the upper 10% SAR responses")
    """
    c1 = fit.slope2 / fit.slope1 if fit.slope1 != 0 else np.nan
    c2, c3 = fit.slope1, fit.intercept1
    c4, c5 = fit.slope2, fit.intercept2
    c6 = fit.x[fit.breakpoint_idx]
    c7 = fit.y[fit.breakpoint_idx]

    c8 = float(np.max(sigma_dB_prime_full) - np.min(sigma_dB_prime_full))

    k = max(1, int(np.ceil(0.10 * len(sigma_dB_prime_full))))
    top_k = np.sort(sigma_dB_prime_full)[-k:]
    c9 = float(np.mean(top_k))

    return np.array([c1, c2, c3, c4, c5, c6, c7, c8, c9], dtype=float)


def pixel_features(sigma_dB_series: np.ndarray, min_segment: int = 5) -> tuple[np.ndarray, PiecewiseFit]:
    """
    End-to-end feature construction for one pixel's historical seasonal SAR
    series (in dB, NOT yet sign-inverted). Returns (C1..C9, piecewise_fit).
    """
    sigma_prime = -np.asarray(sigma_dB_series, dtype=float)  # eq. 2
    x_sorted, F = empirical_cdf_sorted(sigma_prime)           # eq. 3-4
    y = gumbel_transform(F)                                   # eq. 5
    fit = fit_piecewise_linear(x_sorted, y, min_segment=min_segment)
    descriptors = compute_descriptors(fit, sigma_prime)
    return descriptors, fit