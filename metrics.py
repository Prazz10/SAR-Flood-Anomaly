"""
Accuracy assessment following Section 3.4 of Ta et al. (2026).

3.4.1: threshold-independent evaluation of TFAM as a continuous anomaly
score -- ROC-AUC (eq. 10-11) and Average Precision (eq. 12-13).

3.4.2: threshold-swept evaluation of Binary Flood Extent Maps derived from
TFAM (eq. 14) -- F1 (eq. 15), IoU (eq. 16), and Cohen's Kappa, with the
per-event optimal threshold chosen by maximizing F1 (eq. 17-ish, "argmax_t").
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, cohen_kappa_score


@dataclass
class BFEMResult:
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    iou: float
    kappa: float


def continuous_scores(tfam_flat: np.ndarray, ref_flat: np.ndarray) -> dict:
    """
    TFAM discrimination metrics (Sec 3.4.1). tfam_flat and ref_flat must be
    the SAME LENGTH, covering every pixel in the validation boundary
    (pixels outside the TabPFN mask keep their TFAM value = 0, per the
    paper's evaluation protocol -- they are NOT excluded).
    """
    auc = roc_auc_score(ref_flat, tfam_flat)
    ap = average_precision_score(ref_flat, tfam_flat)
    return {"roc_auc": float(auc), "average_precision": float(ap)}


def _confusion(pred: np.ndarray, ref: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((pred == 1) & (ref == 1)))
    fp = int(np.sum((pred == 1) & (ref == 0)))
    fn = int(np.sum((pred == 0) & (ref == 1)))
    tn = int(np.sum((pred == 0) & (ref == 0)))
    return tp, fp, fn, tn


def bfem_at_threshold(tfam_flat: np.ndarray, ref_flat: np.ndarray, t: float) -> BFEMResult:
    pred = (tfam_flat >= t).astype(np.uint8)
    tp, fp, fn, tn = _confusion(pred, ref_flat)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    kappa = cohen_kappa_score(ref_flat, pred)
    return BFEMResult(t, tp, fp, fn, tn, precision, recall, f1, iou, float(kappa))


def sweep_thresholds(tfam_flat: np.ndarray, ref_flat: np.ndarray,
                      t_min: float = 0.05, t_max: float = 0.95, t_step: float = 0.01
                      ) -> list[BFEMResult]:
    """Threshold sensitivity analysis (Sec 3.4.2): sweep t in [0.05, 0.95] step 0.01."""
    thresholds = np.arange(t_min, t_max + 1e-9, t_step)
    return [bfem_at_threshold(tfam_flat, ref_flat, float(t)) for t in thresholds]


def best_f1_result(results: list[BFEMResult]) -> BFEMResult:
    """Event-specific optimal threshold: argmax_t F1(t) (Sec 3.4.2)."""
    return max(results, key=lambda r: r.f1)


def full_event_report(tfam_flat: np.ndarray, ref_flat: np.ndarray) -> dict:
    """
    One-call convenience matching the paper's Table 3 layout: continuous
    scores (AUC, AP) plus the event-specific optimal-threshold BFEM metrics
    (threshold, F1, IoU, Kappa, and the raw confusion counts).
    """
    cont = continuous_scores(tfam_flat, ref_flat)
    sweep = sweep_thresholds(tfam_flat, ref_flat)
    best = best_f1_result(sweep)
    return {
        **cont,
        "optimal_threshold": best.threshold,
        "tp": best.tp, "fp": best.fp, "fn": best.fn, "tn": best.tn,
        "precision": best.precision, "recall": best.recall,
        "f1": best.f1, "iou": best.iou, "kappa": best.kappa,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Convenience wrapper: compute standard accuracy metrics from integer
    ground-truth and prediction arrays (0 = non-flood, 1 = flood).

    Returns dict with keys: IoU, F1, Precision, Recall, OA, Kappa.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    tp, fp, fn, tn = _confusion(y_pred, y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    oa = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    kappa = float(cohen_kappa_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0

    return {
        "IoU": iou,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "OA": oa,
        "Kappa": kappa,
    }