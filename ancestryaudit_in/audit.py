"""
audit.py — Core ancestry-linked performance gap detection.

Trains a model on source (Western) data, evaluates on both held-out
source and target (Asian) populations, and bootstraps the gap for
CI and p-value estimation.
"""
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats


def compute_audit(model, X_source, y_source, X_target, y_target,
                  n_bootstrap=1000, random_state=42):
    """
    Compute ancestry-linked performance gap.

    Parameters
    ----------
    model : sklearn-compatible estimator (unfitted)
    X_source : array-like, shape (n, p)
    y_source : array-like
    X_target : array-like, shape (m, p)
    y_target : array-like
    n_bootstrap : int
    random_state : int

    Returns
    -------
    dict with keys: gap_pp, p_value, cohen_d, ci_95, source_accuracy,
                    target_accuracy, n_source, n_target, trained_model,
                    boot_gaps
    """
    X_source = _to_numpy(X_source)
    X_target = _to_numpy(X_target)
    y_source = np.array(y_source)
    y_target = np.array(y_target)

    # Train on 75% of source; evaluate on 25% hold-out
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_source, y_source,
        test_size=0.25,
        random_state=random_state,
        stratify=y_source
    )

    m = clone(model)
    m.fit(X_tr, y_tr)

    y_pred_source = m.predict(X_te)
    y_pred_target = m.predict(X_target)

    source_acc = float(accuracy_score(y_te, y_pred_source))
    target_acc = float(accuracy_score(y_target, y_pred_target))
    gap_pp     = (source_acc - target_acc) * 100.0

    # Bootstrap CI and p-value on the gap
    rng = np.random.RandomState(random_state)
    boot_gaps = []

    for _ in range(n_bootstrap):
        idx_s = rng.choice(len(y_te),     len(y_te),     replace=True)
        idx_t = rng.choice(len(y_target), len(y_target), replace=True)

        acc_s = accuracy_score(y_te[idx_s],     y_pred_source[idx_s])
        acc_t = accuracy_score(y_target[idx_t], y_pred_target[idx_t])
        boot_gaps.append((acc_s - acc_t) * 100.0)

    boot_gaps  = np.array(boot_gaps)
    ci_low, ci_high = np.percentile(boot_gaps, [2.5, 97.5])

    # t-test on bootstrap distribution vs null of 0
    t_stat, p_value = stats.ttest_1samp(boot_gaps, popmean=0)

    # Cohen's d: mean gap / SD of gap (algorithm-level convention)
    cohen_d = float(np.mean(boot_gaps) / np.std(boot_gaps, ddof=1)) \
              if np.std(boot_gaps, ddof=1) > 0 else 0.0

    return {
        "gap_pp":          float(gap_pp),
        "p_value":         float(p_value),
        "cohen_d":         cohen_d,
        "ci_95":           (float(ci_low), float(ci_high)),
        "source_accuracy": source_acc,
        "target_accuracy": target_acc,
        "n_source":        int(len(X_source)),
        "n_target":        int(len(X_target)),
        "trained_model":   m,
        "boot_gaps":       boot_gaps.tolist(),
    }


def _to_numpy(X):
    if hasattr(X, "values"):
        return X.values.astype(float)
    return np.array(X, dtype=float)
