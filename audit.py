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

  Multiple comparisons:
  When compute_audit() is run once per algorithm (see
  run_per_algorithm_audit() below), the resulting family of p-values is
  corrected with a Holm-Bonferroni step-down procedure before any
  significance claim is made. This is what "Holm correction for multiple
  comparisons ... applied across algorithms" in the Methods section refers
  to — previously that claim was not backed by any code in this file.

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


def holm_bonferroni(p_values, alpha=0.05):
    """
    Holm-Bonferroni step-down correction for a family of hypothesis tests.

    Standard textbook procedure (Holm, 1979):
      1. Sort p-values ascending: p_(1) <= p_(2) <= ... <= p_(n)
      2. Compare p_(i) to alpha / (n - i + 1)
      3. Reject H_(1)...H_(k) where k is the largest index such that
         p_(i) <= alpha / (n - i + 1) for all i <= k (step-down: stop at
         the first failure)
      4. Adjusted p-values are reported as the smallest alpha at which each
         hypothesis would still be rejected, enforced to be monotonically
         non-decreasing in sorted order (hence the running max below).

    Parameters
    ----------
    p_values : array-like of float
        Raw two-sided p-values, one per algorithm/hypothesis, in any order.
    alpha : float
        Family-wise error rate to control. Default 0.05.

    Returns
    -------
    adj_p : np.ndarray, same order as input
        Holm-adjusted p-values. Compare directly to `alpha`.
    reject : np.ndarray of bool, same order as input
        True where the corrected test rejects H0 (i.e. adj_p <= alpha AND
        the step-down chain up to that point also rejected).
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adj_sorted = np.empty(n)
    running_max = 0.0
    for i in range(n):
        raw_adj = (n - i) * sorted_p[i]
        running_max = max(running_max, raw_adj)  # enforce monotonicity
        adj_sorted[i] = min(running_max, 1.0)

    reject_sorted = np.zeros(n, dtype=bool)
    still_rejecting = True
    for i in range(n):
        if still_rejecting and sorted_p[i] <= alpha / (n - i):
            reject_sorted[i] = True
        else:
            still_rejecting = False  # step-down: stop at first non-reject

    adj_p = np.empty(n)
    reject = np.empty(n, dtype=bool)
    adj_p[order] = adj_sorted
    reject[order] = reject_sorted
    return adj_p, reject


def run_per_algorithm_audit(
    base_algorithms, X_source, y_source, X_target, y_target,
    metric="accuracy", n_permutations=1000, random_state=42, alpha=0.05
):
    """
    Run compute_audit() (permutation test) once per algorithm, then apply a
    Holm-Bonferroni correction across the family of algorithm-level tests,
    instead of pooling algorithm-level point estimates into a single
    t-test/bootstrap-CI (statistically invalid: algorithms are not an i.i.d.
    random sample, and a handful of algorithms is far too small for those
    tests' assumptions anyway).

    Significance is determined by Holm-adjusted p-value <= alpha, not raw
    p < 0.05. The test is two-sided: compute_audit() already returns a
    two-sided permutation p-value, and no additional direction filter
    (e.g. requiring gap_pp > 0) is applied here, since that would silently
    convert the test into a one-sided "source > target" test inconsistent
    with the two-sided permutation null described in Methods.

    X_source/y_source should be the FULL (unsplit) source cohort -
    compute_audit() performs its own internal 75/25 train/test split.

    Returns
    -------
    (results, n_significant) : (dict[str, dict], int)
        results[algo_name] is compute_audit()'s full result dict, plus
        'p_holm' (Holm-adjusted p-value) and 'significant_holm' (bool).
        n_significant counts Holm-corrected rejections.
    """
    names = list(base_algorithms.keys())
    raw_results = {}
    raw_p = []

    for name in names:
        r = compute_audit(
            clone(base_algorithms[name]), X_source, y_source, X_target, y_target,
            n_permutations=n_permutations, random_state=random_state,
            metric=metric
        )
        raw_results[name] = r
        raw_p.append(r["p_value"])

    adj_p, reject = holm_bonferroni(raw_p, alpha=alpha)

    print(f"{'Algo':<5}{'Gap(pp)':>10}{'raw p':>10}{'Holm p':>10}{'Significant':>13}")
    print("-" * 51)
    n_significant = 0
    for name, ap, rej in zip(names, adj_p, reject):
        r = raw_results[name]
        r["p_holm"] = float(ap)
        r["significant_holm"] = bool(rej)
        n_significant += int(rej)
        print(f"{name:<5}{r['gap_pp']:>+10.2f}{r['p_value']:>10.4f}"
              f"{ap:>10.4f}{'YES' if rej else 'no':>13}")

    print(f"\n{n_significant}/{len(names)} algorithms significant after Holm "
          f"correction (metric={metric}, family size={len(names)})")

    return raw_results, n_significant


def _to_numpy(X):
    if hasattr(X, "values"):
        return X.values.astype(float)
    return np.array(X, dtype=float)
