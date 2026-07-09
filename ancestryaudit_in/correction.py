"""
correction.py — Supervised fine-tuning correction.

Combines source data with n_samples labeled target instances,
retrains the model, and validates robustness across 10 random seeds.
"""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from scipy import stats

DEFAULT_SEEDS = [42, 0, 1, 7, 13, 17, 21, 99, 123, 256]


def apply_correction(model, X_source, y_source,
                     X_target_labeled, y_target_labeled,
                     n_samples=75, random_state=42, seeds=None):
    """
    Supervised fine-tuning: source data + n_samples from labeled target.

    For each seed:
      1. Stratified sample of n_samples from target → fine-tune set.
      2. Remaining target → holdout (never seen during training).
      3. Train baseline (source only) and fine-tuned (source + samples).
      4. Compute delta = acc_finetuned - acc_baseline on holdout.

    Parameters
    ----------
    model : sklearn-compatible estimator
    X_source, y_source : source (Western) training data
    X_target_labeled, y_target_labeled : labeled target data
    n_samples : int, target samples to include per seed run
    random_state : int, primary seed (also first entry of seeds list)
    seeds : list of int, seeds for robustness testing

    Returns
    -------
    best_model : fitted corrected model (from primary seed)
    results : dict with delta_pp, p_value, n_used, seed_robustness,
              baseline_accuracy, corrected_accuracy, all_positive
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    X_source = _to_numpy(X_source)
    X_target = _to_numpy(X_target_labeled)
    y_source = np.array(y_source)
    y_target = np.array(y_target_labeled)

    n_actual = min(n_samples, len(y_target))
    seed_deltas   = []
    seed_models   = []

    for seed in seeds:
        rng = np.random.RandomState(seed)

        # Stratified sample
        idx_0 = np.where(y_target == 0)[0]
        idx_1 = np.where(y_target == 1)[0]
        n0 = min(int(round(n_actual * len(idx_0) / len(y_target))), len(idx_0))
        n1 = min(n_actual - n0, len(idx_1))

        if n0 == 0 or n1 == 0:
            continue

        sel  = np.concatenate([rng.choice(idx_0, n0, replace=False),
                                rng.choice(idx_1, n1, replace=False)])
        hold = np.setdiff1d(np.arange(len(y_target)), sel)

        if len(hold) == 0 or len(np.unique(y_target[hold])) < 2:
            continue

        X_comb = np.vstack([X_source, X_target[sel]])
        y_comb  = np.concatenate([y_source, y_target[sel]])

        # Baseline
        m_base = clone(model)
        m_base.fit(X_source, y_source)
        acc_base = accuracy_score(y_target[hold], m_base.predict(X_target[hold]))

        # Fine-tuned
        m_ft = clone(model)
        m_ft.fit(X_comb, y_comb)
        acc_ft = accuracy_score(y_target[hold], m_ft.predict(X_target[hold]))

        seed_deltas.append((acc_ft - acc_base) * 100.0)
        seed_models.append(m_ft)

    if not seed_deltas:
        raise ValueError(
            "No valid seeds produced holdout sets. "
            "Check that n_samples < len(X_target_labeled) and "
            "both classes are present in the target data."
        )

    seed_deltas = np.array(seed_deltas)
    mean_delta  = float(np.mean(seed_deltas))

    t_stat, p_value = stats.ttest_1samp(seed_deltas, popmean=0)

    # Overall baseline (source-only on full target)
    m_ref = clone(model)
    m_ref.fit(X_source, y_source)
    baseline_acc    = float(accuracy_score(y_target, m_ref.predict(X_target)))
    corrected_acc   = baseline_acc + mean_delta / 100.0

    best_model = seed_models[0]  # primary seed = index 0

    results = {
        "delta_pp":   mean_delta,
        "p_value":    float(p_value),
        "n_used":     int(n_actual),
        "seed_robustness": {
            "mean":       float(np.mean(seed_deltas)),
            "sd":         float(np.std(seed_deltas, ddof=1)),
            "min":        float(np.min(seed_deltas)),
            "max":        float(np.max(seed_deltas)),
            "all_positive": bool(np.all(seed_deltas > 0)),
            "n_positive": int(np.sum(seed_deltas > 0)),
            "n_seeds":    int(len(seed_deltas)),
        },
        "baseline_accuracy":  baseline_acc,
        "corrected_accuracy": corrected_acc,
        "all_positive":       bool(np.all(seed_deltas > 0)),
    }

    return best_model, results


def _to_numpy(X):
    if hasattr(X, "values"):
        return X.values.astype(float)
    return np.array(X, dtype=float)
