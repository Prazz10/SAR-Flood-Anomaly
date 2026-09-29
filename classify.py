"""
TabPFN-based flood classification, following Section 3.3 of Ta et al. (2026).

The paper uses TabPFN purely for in-context inference: a small set of
hand-labeled points (83 samples: 44 flood / 39 non-flood) supplies the C1-C9
feature vectors as "context examples", and TabPFN then classifies every
other pixel's C1-C9 vector without any gradient-based training. A fixed
probability threshold of 0.5 turns P(flood) into the binary maximum
flood-extent mask (eq. 8).

Requires the `tabpfn` package and a HuggingFace account with the
Prior-Labs/tabpfn model terms accepted (see README "Setup" section) -- this
cannot run inside this sandbox (huggingface.co is not reachable here), but
will run on your own machine once you've accepted the model license.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class FloodClassifier:
    """Thin wrapper matching the paper's TabPFN configuration (Sec 3.3)."""
    n_estimators: int = 8
    random_state: int = 0
    device: str = "cpu"

    def __post_init__(self):
        from tabpfn import TabPFNClassifier  # deferred import
        self._clf = TabPFNClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            device=self.device,
        )
        self._fitted = False

    def fit_context(self, X_context: np.ndarray, y_context: np.ndarray):
        """
        X_context: (n_samples, 9) array of C1-C9 for hand-labeled points
        y_context: (n_samples,) array, 1 = flood, 0 = non-flood (already
                   recoded per the paper: raw label 2->1, 1->0)
        """
        self._clf.fit(X_context, y_context)
        self._fitted = True
        return self

    def predict_proba_flood(self, X: np.ndarray) -> np.ndarray:
        """Return P(flood) for each row of X (eq. 8: P_f = P(y=1 | X_i))."""
        if not self._fitted:
            raise RuntimeError("call fit_context() first")
        proba = self._clf.predict_proba(X)
        # class order from sklearn-style API: locate the column for class 1
        classes = list(self._clf.classes_)
        return proba[:, classes.index(1)]

    def max_flood_extent_mask(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary mask via the paper's fixed threshold of 0.5 (Sec 3.3)."""
        return (self.predict_proba_flood(X) >= threshold).astype(np.uint8)


def recode_labels(raw_labels: np.ndarray) -> np.ndarray:
    """
    Paper's labeling convention (Sec 3.3): stored as value=1 (non-flood),
    value=2 (flood); recoded to y=0 (non-flood), y=1 (flood).
    """
    raw = np.asarray(raw_labels)
    y = np.zeros_like(raw)
    y[raw == 2] = 1
    y[raw == 1] = 0
    return y