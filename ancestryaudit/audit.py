"""
audit.py — Core ancestry-linked performance gap detection.

Statistical design:
  Null hypothesis H0: source and target are drawn from the same
  performance distribution (no true ancestry-linked gap).

  Test: label-permutation test on per-sample correctness.
  Permute which samples are "source test" vs "target" thousands of times,
  recompute the gap under each permuted null, and locate the observed
  gap in that distribution.

  This is the textbook-correct null for this problem.
  It does NOT test whether a bootstrapped point estimate excludes zero
  (which would be circular and produce near-zero p-values by construction
  on any finite dataset, as the original implementation did).

Cohen's d:
  Proper between-group effect size on raw per-sample correctness
  (binary 0/1), using pooled within-group SD.
  Independent of the permutation array.
"""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def compute_audit(model, X_source, y_source, X_target, y_target,
                  n_permutations=1000, random_state=42):
    """
    Compute ancestry-linked performance gap with permutation-based inference.

    Parameters
    ----------
    model : sklearn-compatible estimator (unfitted)
    X_source : array-like, shape (n, p)
    y_source : array-like
    X_target : array-like, shape (m, p)
    y_target : array-like
    n_permutations : int, permutation test iterations
    random_state : int

    Returns
    -------
    dict with keys:
        gap_pp              — observed gap in percentage points
        p_value             — permutation p-value (two-sided)
        cohen_d             — between-group effect size (independent of p-value)
        null_ci             — 2.5/97.5 percentiles of permutation null distribution
                              use for reference of null distribution spread)
        source_accuracy     — model accuracy on held-out source
        target_accuracy     — model accuracy on target
        n_source, n_target
        trained_model
        perm_gaps           — full permutation null distribution
        null_mean           — mean of null distribution (should be near 0)
        null_sd             — SD of null distribution
    """
    X_source = _to_numpy(X_source)
    X_target = _to_numpy(X_target)
    y_source = np.array(y_source)
    y_target = np.array(y_target)

    # Train on 75% of source; evaluate on held-out 25%
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

    # ── Per-sample correctness arrays ─────────────────────────────────────────
    correct_s = (y_pred_source == y_te).astype(float)      # shape (n_te,)
    correct_t = (y_pred_target == y_target).astype(float)  # shape (n_tgt,)

    n_s = len(correct_s)
    n_t = len(correct_t)

    # ── Permutation test ──────────────────────────────────────────────────────
    # Pool both correctness arrays; repeatedly shuffle; recompute gap.
    # This constructs the null distribution of the gap under H0:
    # "source and target are exchangeable" — the correct null for this problem.
    pooled = np.concatenate([correct_s, correct_t])
    rng    = np.random.RandomState(random_state)

    perm_gaps = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled      = rng.permutation(pooled)
        perm_gaps[i]  = (shuffled[:n_s].mean() - shuffled[n_s:].mean()) * 100.0

    # Two-sided p-value: proportion of null gaps at least as extreme as observed
    p_value = float(np.mean(np.abs(perm_gaps) >= np.abs(gap_pp)))
    # Floor at 1/n_permutations to avoid reporting p=0.0000
    p_value = max(p_value, 1.0 / n_permutations)

    null_ci = np.percentile(perm_gaps, [2.5, 97.5])

    # ── Cohen's d (proper between-group effect size) ───────────────────────────
    # Independent of the permutation array.
    # d = (mean_source_correct - mean_target_correct) / pooled_within_SD
    m1, m2 = correct_s.mean(), correct_t.mean()
    s1, s2 = correct_s.std(ddof=1), correct_t.std(ddof=1)
    pooled_sd = np.sqrt(
        ((n_s - 1) * s1**2 + (n_t - 1) * s2**2) / (n_s + n_t - 2)
    )
    cohen_d = float((m1 - m2) / pooled_sd) if pooled_sd > 0 else 0.0

    return {
        "gap_pp":          gap_pp,
        "p_value":         p_value,
        "cohen_d":         cohen_d,
        "null_ci":         (float(null_ci[0]), float(null_ci[1])),
        "source_accuracy": source_acc,
        "target_accuracy": target_acc,
        "n_source":        int(len(X_source)),
        "n_target":        int(len(X_target)),
        "trained_model":   m,
        "perm_gaps":       perm_gaps.tolist(),
        "null_mean":       float(perm_gaps.mean()),
        "null_sd":         float(perm_gaps.std(ddof=1)),
    }


def _to_numpy(X):
    if hasattr(X, "values"):
        return X.values.astype(float)
    return np.array(X, dtype=float)
