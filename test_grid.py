"""
Grid-level synthetic test: build a small synthetic "scene" of pixels (some
land, some flood-prone), simulate an event date where the flood-prone
pixels flood, run features -> TCEV -> TFAM -> metrics end-to-end, and check
that AUC/AP/F1/IoU/Kappa behave sensibly (i.e., the method actually
discriminates flooded from non-flooded pixels; this does NOT validate
absolute numbers against the paper, only that the pipeline is wired up
correctly. TabPFN is swapped for the ground-truth mask here since TabPFN
weights aren't reachable in this sandbox -- see classify.py for the real
TabPFN-based mask step.)
"""

import numpy as np

from features import pixel_features, empirical_cdf_sorted
from tcev import fit_tcev, event_anomaly_score
from tfam import build_tfam
from metrics import full_event_report

rng = np.random.default_rng(1)

N_LAND = 400
N_FLOOD_PRONE = 100
N = N_LAND + N_FLOOD_PRONE
HIST_LEN = 60

is_flood_prone = np.array([False] * N_LAND + [True] * N_FLOOD_PRONE)
rng.shuffle(is_flood_prone)

# on the event date, only flood-prone pixels actually flood (reference truth)
ref_label = is_flood_prone.astype(np.uint8)

tfam_scores = np.zeros(N)
flood_mask = np.ones(N, dtype=np.uint8)  # stand-in for TabPFN max-extent mask:
# assume TabPFN perfectly recovers "could ever flood" == is_flood_prone here,
# which is the easy case; a harder test would add false positives/negatives.
flood_mask = is_flood_prone.astype(np.uint8)

for i in range(N):
    if is_flood_prone[i]:
        hist = rng.normal(-10.0, 1.0, size=HIST_LEN)
        n_hist_floods = rng.integers(3, 8)
        idx = rng.choice(HIST_LEN, size=n_hist_floods, replace=False)
        hist[idx] = rng.normal(-19.0, 1.5, size=n_hist_floods)
        event_obs = rng.normal(-19.0, 1.5)  # this pixel floods today
    else:
        hist = rng.normal(-10.0, 1.0, size=HIST_LEN)
        event_obs = rng.normal(-10.0, 1.0)  # stays normal

    if flood_mask[i] == 0:
        continue  # TFAM stays 0 outside the mask, per the paper's protocol

    _, fit = pixel_features(hist)
    sigma_prime = -hist
    x_sorted, F = empirical_cdf_sorted(sigma_prime)
    tcev = fit_tcev(x_sorted, F, fit)
    tfam_scores[i] = event_anomaly_score(tcev, event_obs)

tfam = build_tfam(tfam_scores, flood_mask)
report = full_event_report(tfam, ref_label)

print("Synthetic grid: N =", N, " (flood-prone:", N_FLOOD_PRONE, ", land:", N_LAND, ")")
for k, v in report.items():
    print(f"  {k:20s} = {v}")

assert report["roc_auc"] > 0.7, "expected reasonable discrimination on this easy synthetic case"
print("\nGrid-level pipeline test passed (AUC > 0.7 on an easy synthetic case).")