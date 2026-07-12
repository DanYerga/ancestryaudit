"""
test_power_calibration.py
=========================
Regression test: flip_rate calibration table accuracy.

Verifies that the empirical flip_rate → gap_pp mapping used in
power_analysis() is still accurate. Each flip_rate should produce
a mean observed gap within ±2pp of the calibration reference value.

If this test fails, the calibration table in power_analysis() needs
to be re-run against the current compute_audit() implementation.

Reference calibration (n_source=451, n_target=242, n_trials=50):
    flip_rate=0.020  →  mean gap ≈ +1.67pp
    flip_rate=0.050  →  mean gap ≈ +4.20pp
    flip_rate=0.100  →  mean gap ≈ +8.29pp

Run:
    python tests/test_power_calibration.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.linear_model import LogisticRegression
from ancestryaudit.audit import compute_audit

# Reference values from empirical calibration (July 2026)
CALIBRATION_REFERENCE = {
    0.020: 1.67,
    0.050: 4.20,
    0.100: 8.29,
}
TOLERANCE_PP = 2.0    # acceptable deviation from reference
N_TRIALS     = 30     # enough for a stable mean, fast to run
N_SOURCE     = 451
N_TARGET     = 242


def test_flip_rate_calibration():
    """
    Each flip_rate should produce mean gap within ±TOLERANCE_PP
    of the reference calibration value.
    """
    model = LogisticRegression(max_iter=500, random_state=42)
    all_pass = True

    print(f"{'flip_rate':>10}  {'reference':>10}  "
          f"{'observed':>10}  {'deviation':>10}  {'status':>8}")
    print("-" * 55)

    for flip_rate, ref_gap in CALIBRATION_REFERENCE.items():
        rng  = np.random.RandomState(42 + int(flip_rate * 10000))
        gaps = []

        for trial in range(N_TRIALS):
            X_s = rng.randn(N_SOURCE, 50)
            y_s = (X_s[:, 0] + X_s[:, 1] > 0).astype(int)
            X_t = rng.randn(N_TARGET, 50)
            y_t = (X_t[:, 0] + X_t[:, 1] > 0).astype(int)

            n_flip = int(round(N_TARGET * flip_rate))
            if n_flip > 0:
                idx    = rng.choice(N_TARGET, size=n_flip, replace=False)
                y_t[idx] = 1 - y_t[idx]

            r = compute_audit(model, X_s, y_s, X_t, y_t,
                              n_permutations=200, random_state=trial)
            gaps.append(r['gap_pp'])

        mean_gap  = float(np.mean(gaps))
        deviation = abs(mean_gap - ref_gap)
        passed    = deviation <= TOLERANCE_PP
        status    = "PASS ✓" if passed else "FAIL ✗"

        if not passed:
            all_pass = False

        print(f"  {flip_rate:>8.3f}  {ref_gap:>+9.2f}pp  "
              f"{mean_gap:>+9.2f}pp  {deviation:>+9.2f}pp  {status}")

    print()
    if all_pass:
        print("ALL CALIBRATION CHECKS PASSED ✓")
        print("flip_rate → gap_pp mapping is consistent with reference.")
    else:
        print("CALIBRATION REGRESSION DETECTED ✗")
        print("Re-run the full calibration sweep (50 trials, all flip_rates)")
        print("and update CALIB_FLIP / CALIB_GAP arrays in power_analysis().")
        raise AssertionError(
            "Flip-rate calibration has drifted beyond tolerance. "
            "power_analysis() gap estimates are no longer reliable."
        )

    return all_pass


if __name__ == "__main__":
    test_flip_rate_calibration()
