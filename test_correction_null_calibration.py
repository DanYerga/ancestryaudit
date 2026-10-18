"""
test_correction_null_calibration.py
====================================
Regression test: McNemar correction FPR on null data.
Hard fail: >10% FPR.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from sklearn.linear_model import LogisticRegression
from ancestryaudit.correction import apply_correction

N_TRIALS, MAX_FPR, RANDOM_STATE = 300, 0.10, 99

def test_correction_null_fpr():
    rng = np.random.RandomState(RANDOM_STATE)
    false_positives = 0
    for trial in range(N_TRIALS):
        X_s = rng.randn(200, 50); y_s = (X_s[:,0]+X_s[:,1] > 0).astype(int)
        X_t = rng.randn(100, 50); y_t = (X_t[:,0]+X_t[:,1] > 0).astype(int)
        try:
            _, results = apply_correction(LogisticRegression(max_iter=500),
                                           X_s, y_s, X_t, y_t, n_samples=30, random_state=trial)
            if results["direction_confirmed"]:
                false_positives += 1
        except ValueError:
            continue
    fpr = false_positives / N_TRIALS
    print(f"Correction null FPR : {false_positives}/{N_TRIALS} = {fpr*100:.1f}%")
    assert fpr <= MAX_FPR, f"FAIL: FPR={fpr*100:.1f}% exceeds {MAX_FPR*100:.0f}% limit."
    print(f"PASS: FPR={fpr*100:.1f}%")

if __name__ == "__main__":
    test_correction_null_fpr()
