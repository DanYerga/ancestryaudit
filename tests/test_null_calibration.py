"""
test_null_calibration.py
========================
Regression test: permutation test false positive rate on null data.

Under H0 (source and target from identical distributions),
the test should flag "correction_required" approximately α=5%
of the time at threshold_pp=2.0, threshold_p=0.05.

Acceptable range: 0–15% for n_trials=20 (sampling noise expected).
Hard fail: >20% (indicates broken null — the original bootstrap bug).

Run:
    python tests/test_null_calibration.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from ancestryaudit import AncestryAuditFramework
from sklearn.linear_model import LogisticRegression

N_TRIALS    = 20
ALPHA       = 0.05
THRESHOLD_PP = 2.0
MAX_ACCEPTABLE_FPR = 0.20   # hard fail above this


def test_null_fpr():
    """FPR on identical-distribution data should be near 5%, not near 85%."""
    rng = np.random.RandomState(0)
    false_positives = 0

    for trial in range(N_TRIALS):
        X_s = rng.randn(200, 50)
        y_s = (X_s[:, 0] + X_s[:, 1] > 0).astype(int)
        X_t = rng.randn(80, 50)
        y_t = (X_t[:, 0] + X_t[:, 1] > 0).astype(int)

        fw = AncestryAuditFramework(
            random_state=trial,
            threshold_pp=THRESHOLD_PP,
            threshold_p=ALPHA,
            n_bootstrap=500
        )
        from ancestryaudit.audit import compute_audit
        r = compute_audit(
            LogisticRegression(max_iter=500),
            X_s, y_s, X_t, y_t,
            n_permutations=500,
            random_state=trial
        )
        if r['p_value'] < ALPHA and abs(r['gap_pp']) > THRESHOLD_PP:
            false_positives += 1

    fpr = false_positives / N_TRIALS
    print(f"Null FPR: {false_positives}/{N_TRIALS} = {fpr*100:.1f}%")
    print(f"Target  : ~5%  |  Hard limit: {MAX_ACCEPTABLE_FPR*100:.0f}%")

    assert fpr <= MAX_ACCEPTABLE_FPR, (
        f"FAIL: FPR={fpr*100:.1f}% exceeds {MAX_ACCEPTABLE_FPR*100:.0f}% limit. "
        f"This indicates the original circular-bootstrap bug has regressed. "
        f"Check compute_audit() uses permutation test, not bootstrap t-test."
    )
    print(f"PASS ✓  FPR={fpr*100:.1f}% within acceptable range.")
    return fpr


if __name__ == "__main__":
    test_null_fpr()
