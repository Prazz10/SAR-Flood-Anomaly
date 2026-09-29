"""
Synthetic validation of features.py + tcev.py.

We simulate per-pixel historical SAR (VH, dB) series for two archetypes:
  - "land" pixel: normally distributed backscatter, no flood history
  - "flood-prone" pixel: mostly normal, but with occasional historical flood
    dips (much lower dB, since water = low backscatter) -- this bimodality
    is exactly what the piecewise fit / TCEV second component should pick up

Then we simulate an "event" observation for each pixel: a land pixel stays
in its normal range, a flood-prone pixel gets a strong dip (flood event).
We check that:
  1. the piecewise fit finds a later, steeper second segment for flood-prone
     pixels (i.e. C1 = slope2/slope1 > 1, indicating heavier tail),
  2. the TCEV-based anomaly score (TFAM value) is much higher for the
     flooded event observation than for the non-flooded one.
"""

import numpy as np

from features import pixel_features
from tcev import fit_tcev, event_anomaly_score
from features import empirical_cdf_sorted, gumbel_transform

rng = np.random.default_rng(0)


def make_land_series(n=70, mean=-10.0, std=1.0):
    return rng.normal(mean, std, size=n)


def make_flood_prone_series(n=70, mean=-10.0, std=1.0, n_floods=6, flood_mean=-19.0, flood_std=1.5):
    s = rng.normal(mean, std, size=n)
    flood_idx = rng.choice(n, size=n_floods, replace=False)
    s[flood_idx] = rng.normal(flood_mean, flood_std, size=n_floods)
    return s


def run_pixel(sigma_dB_hist, sigma_dB_event, label):
    descriptors, fit = pixel_features(sigma_dB_hist)
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = descriptors

    sigma_prime = -sigma_dB_hist
    x_sorted, F = empirical_cdf_sorted(sigma_prime)
    tcev = fit_tcev(x_sorted, F, fit)
    score = event_anomaly_score(tcev, sigma_dB_event)

    print(f"--- {label} ---")
    print(f"  C1 (slope ratio)      = {c1:8.3f}")
    print(f"  C2,C3 (regular seg)   = {c2:8.3f}, {c3:8.3f}")
    print(f"  C4,C5 (extreme seg)   = {c4:8.3f}, {c5:8.3f}")
    print(f"  C6,C7 (breakpoint)    = {c6:8.3f}, {c7:8.3f}")
    print(f"  C8 (amplitude)        = {c8:8.3f}")
    print(f"  C9 (upper-10% mean)   = {c9:8.3f}")
    print(f"  TCEV params           = a1={tcev.a1:.3f} b1={tcev.b1:.3f} "
          f"a2={tcev.a2:.3f} b2={tcev.b2:.3f}")
    print(f"  event obs sigma_dB    = {sigma_dB_event:.2f}")
    print(f"  TFAM anomaly score    = {score:.4f}\n")
    return descriptors, score


if __name__ == "__main__":
    print("=" * 60)
    print("Pixel A: LAND (no flood history), event = normal observation")
    land_hist = make_land_series()
    land_event = rng.normal(-10.0, 1.0)  # normal, non-flood event obs
    d_land, s_land = run_pixel(land_hist, land_event, "Land pixel, normal event")

    print("=" * 60)
    print("Pixel B: FLOOD-PRONE (has flood history), event = flood observation")
    flood_hist = make_flood_prone_series()
    flood_event = rng.normal(-19.0, 1.0)  # simulated flood dip on event date
    d_flood, s_flood = run_pixel(flood_hist, flood_event, "Flood-prone pixel, flood event")

    print("=" * 60)
    print("Pixel C: FLOOD-PRONE pixel, but event date happens to be NORMAL")
    flood_event_normal = rng.normal(-10.0, 1.0)
    d_flood2, s_flood2 = run_pixel(flood_hist, flood_event_normal,
                                    "Flood-prone pixel, non-flood event date")

    print("=" * 60)
    print("SANITY CHECKS")
    print(f"  slope ratio C1: land={d_land[0]:.2f}  flood-prone={d_flood[0]:.2f}  "
          f"(expect flood-prone > land if second-population tail is heavier)")
    print(f"  TFAM score:  land/normal-event={s_land:.4f}  "
          f"flood-prone/flood-event={s_flood:.4f}  "
          f"flood-prone/normal-event={s_flood2:.4f}")
    assert s_flood > s_land, "flood event on flood-prone pixel should score higher than normal event on land pixel"
    assert s_flood > s_flood2, "flood event should score higher than a normal event on the SAME pixel"
    print("\nAll sanity checks passed.")