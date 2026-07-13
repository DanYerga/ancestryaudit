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
                  n_permutations=1000, random_state=42, metric="accuracy"):
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
    metric : {"accuracy", "balanced_accuracy"}, default "accuracy"
        "accuracy" reproduces the original behavior exactly (backward
        compatible - existing calibration tests depend on this being
        unchanged). "balanced_accuracy" is robust to differing class
        priors between source and target: use this whenever the two
        groups may have different class balance, since a raw accuracy
        gap can otherwise reflect prior mismatch rather than genuine
        signal. Permutation for balanced_accuracy shuffles (true, pred)
        label pairs between groups (not a simple correctness-array
        shuffle), since balanced accuracy is class-conditional, not a
        simple mean. cohen_d is only defined for metric="accuracy".

    Returns
    -------
    dict with keys:
        gap_pp              — observed gap in percentage points
        p_value             — permutation p-value (two-sided)
        cohen_d             — between-group effect size (accuracy metric
                              only; None for balanced_accuracy)
        null_ci             — 2.5/97.5 percentiles of permutation null distribution
                              use for reference of null distribution spread)
        source_accuracy     — model score (per `metric`) on held-out source
        target_accuracy     — model score (per `metric`) on target
        n_source, n_target
        trained_model
        perm_gaps           — full permutation null distribution
        null_mean           — mean of null distribution (should be near 0)
        null_sd             — SD of null distribution
        metric              — which metric was used
    """
    if metric not in ("accuracy", "balanced_accuracy"):
        raise ValueError(
            f"metric must be 'accuracy' or 'balanced_accuracy', got {metric!r}")

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

    if metric == "accuracy":
        source_acc = float(accuracy_score(y_te, y_pred_source))
        target_acc = float(accuracy_score(y_target, y_pred_target))
        gap_pp     = (source_acc - target_acc) * 100.0

        # ── Per-sample correctness arrays ───────────────────────────────────
        correct_s = (y_pred_source == y_te).astype(float)      # shape (n_te,)
        correct_t = (y_pred_target == y_target).astype(float)  # shape (n_tgt,)

        n_s = len(correct_s)
        n_t = len(correct_t)

        # ── Permutation test ────────────────────────────────────────────────
        # Pool both correctness arrays; repeatedly shuffle; recompute gap.
        # This constructs the null distribution of the gap under H0:
        # "source and target are exchangeable" — the correct null for this.
        pooled = np.concatenate([correct_s, correct_t])
        rng    = np.random.RandomState(random_state)

        perm_gaps = np.empty(n_permutations)
        for i in range(n_permutations):
            shuffled      = rng.permutation(pooled)
            perm_gaps[i]  = (shuffled[:n_s].mean() - shuffled[n_s:].mean()) * 100.0

        # ── Cohen's d (proper between-group effect size) ───────────────────
        m1, m2 = correct_s.mean(), correct_t.mean()
        s1, s2 = correct_s.std(ddof=1), correct_t.std(ddof=1)
        pooled_sd = np.sqrt(
            ((n_s - 1) * s1**2 + (n_t - 1) * s2**2) / (n_s + n_t - 2)
        )
        cohen_d = float((m1 - m2) / pooled_sd) if pooled_sd > 0 else 0.0

    else:  # metric == "balanced_accuracy"
        from sklearn.metrics import balanced_accuracy_score
        source_acc = float(balanced_accuracy_score(y_te, y_pred_source))
        target_acc = float(balanced_accuracy_score(y_target, y_pred_target))
        gap_pp     = (source_acc - target_acc) * 100.0

        n_s = len(y_te)
        n_t = len(y_target)

        # Balanced accuracy is class-conditional, not a simple mean, so
        # permute (true, predicted) pairs between groups, not just a 0/1
        # correctness array.
        true_pooled = np.concatenate([y_te, y_target])
        pred_pooled = np.concatenate([y_pred_source, y_pred_target])
        idx = np.arange(n_s + n_t)
        rng = np.random.RandomState(random_state)

        perm_gaps = np.empty(n_permutations)
        for i in range(n_permutations):
            shuffled = rng.permutation(idx)
            g_s, g_t = shuffled[:n_s], shuffled[n_s:]
            perm_gaps[i] = (
                balanced_accuracy_score(true_pooled[g_s], pred_pooled[g_s]) -
                balanced_accuracy_score(true_pooled[g_t], pred_pooled[g_t])
            ) * 100.0

        cohen_d = None  # not a simple mean-based effect size here

    # Two-sided p-value: proportion of null gaps at least as extreme as observed
    p_value = float(np.mean(np.abs(perm_gaps) >= np.abs(gap_pp)))
    # Floor at 1/n_permutations to avoid reporting p=0.0000
    p_value = max(p_value, 1.0 / n_permutations)

    null_ci = np.percentile(perm_gaps, [2.5, 97.5])

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
        "metric":          metric,
    }


def _to_numpy(X):
    if hasattr(X, "values"):
        return X.values.astype(float)
    return np.array(X, dtype=float)
