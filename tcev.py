"""
Two-Component Extreme Value (TCEV) model fitting, following Section 3.2
(eq. 7) and Section 3.3 (eq. 9) of Ta et al. (2026).

    F_TCEV(x) = exp(-exp(-a1*(x-b1))) * exp(-exp(-a2*(x-b2)))     (eq. 7)

a product of two Gumbel CDFs. Initial values for the nonlinear fit come
from the piecewise-linear fit (features.py):
    a1 = C2, b1 = -C3/C2      (regular segment)
    a2 = C4, b2 = -C5/C4      (extreme segment)

The final parameters are estimated per pixel by nonlinear least-squares of
the full TCEV CDF against the empirical CDF of all historical seasonal
observations (not just a piecewise approximation).
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

try:
    from .features import PiecewiseFit
except ImportError:
    from features import PiecewiseFit


@dataclass
class TCEVParams:
    a1: float
    b1: float
    a2: float
    b2: float

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.exp(-np.exp(-self.a1 * (x - self.b1))) * np.exp(
            -np.exp(-self.a2 * (x - self.b2))
        )


def _tcev_cdf(x, a1, b1, a2, b2):
    return np.exp(-np.exp(-a1 * (x - b1))) * np.exp(-np.exp(-a2 * (x - b2)))


def initial_guess_from_piecewise(fit: PiecewiseFit) -> TCEVParams:
    a1 = fit.slope1
    b1 = -fit.intercept1 / fit.slope1 if fit.slope1 != 0 else 0.0
    a2 = fit.slope2
    b2 = -fit.intercept2 / fit.slope2 if fit.slope2 != 0 else 0.0
    return TCEVParams(a1, b1, a2, b2)


def fit_tcev(sigma_dB_prime_sorted: np.ndarray, F_empirical: np.ndarray,
             fit: PiecewiseFit, maxfev: int = 20000) -> TCEVParams:
    """
    Nonlinear least-squares fit of the TCEV CDF to the empirical CDF of the
    complete historical seasonal series for one pixel (eq. 7).
    """
    init = initial_guess_from_piecewise(fit)
    # clamp initial slopes away from 0 / wrong sign -- a degenerate a->0
    # collapses one Gumbel component to a constant and destabilizes the fit
    eps = 1e-3
    a1_0 = init.a1 if init.a1 > eps else eps
    a2_0 = init.a2 if init.a2 > eps else eps
    p0 = [a1_0, init.b1, a2_0, init.b2]
    # keep both slopes strictly positive (Gumbel CDFs must be increasing);
    # intercepts (location params) are left unbounded
    bounds = ([eps, -np.inf, eps, -np.inf], [np.inf, np.inf, np.inf, np.inf])
    try:
        popt, _ = curve_fit(
            _tcev_cdf, sigma_dB_prime_sorted, F_empirical, p0=p0,
            bounds=bounds, maxfev=maxfev
        )
        return TCEVParams(*popt)
    except RuntimeError:
        # fitting failed to converge -- fall back to the piecewise initial
        # values rather than crashing a whole-scene batch job
        return TCEVParams(a1_0, init.b1, a2_0, init.b2)


def event_anomaly_score(tcev: TCEVParams, sigma_dB_event: float) -> float:
    """
    Compute the TFAM anomaly score for one pixel on one acquisition date
    (eq. 9): the event observation's cumulative probability under the
    pixel's fitted historical TCEV distribution, in the same sign-inverted
    domain used for fitting.
    """
    sigma_prime_event = -float(sigma_dB_event)
    return float(tcev.cdf(sigma_prime_event))