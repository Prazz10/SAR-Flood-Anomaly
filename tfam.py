"""
Temporal Flood Anomaly Map (TFAM) construction, following Section 3.3
(final paragraphs) of Ta et al. (2026).

TFAM(x, y) on a given acquisition date =
    F_TCEV(sigma_dB'_event(x, y))   if (x, y) is inside the TabPFN-derived
                                     maximum historical flood-extent mask
    0                                otherwise (including pixels with no
                                     TFAM value at all)

This keeps the spatial classification (TabPFN mask = "could this pixel ever
plausibly flood, historically") and the temporal anomaly score (TCEV CDF =
"how unusual is today's observation for this specific pixel") as two
genuinely separate signals that only get combined at the very end.
"""

import numpy as np


def build_tfam(anomaly_scores: np.ndarray, flood_mask: np.ndarray) -> np.ndarray:
    """
    anomaly_scores: array of F_TCEV(event obs) per pixel, same shape as mask
    flood_mask: binary array, 1 = inside TabPFN max flood extent, 0 = outside

    Returns the TFAM array: anomaly_scores where mask==1, else 0.
    """
    assert anomaly_scores.shape == flood_mask.shape
    tfam = np.zeros_like(anomaly_scores, dtype=float)
    tfam[flood_mask.astype(bool)] = anomaly_scores[flood_mask.astype(bool)]
    return tfam


def binary_flood_map(tfam: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold segmentation of a TFAM into a Binary Flood Extent Map (eq. 14)."""
    return (tfam >= threshold).astype(np.uint8)