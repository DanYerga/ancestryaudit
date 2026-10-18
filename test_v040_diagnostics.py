"""
Tests for the v0.4.0 diagnostics:
  - direction-aware recommendation in audit()
  - check_composition() / diagnose()
  - warning when correct() runs without a confirmed gap
  - audit_many() with Holm correction and direction verdict
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from ancestryaudit import AncestryAuditFramework, check_composition


def _cohort(rng, n, sep, flip=0.0):
    """Two-class Gaussian cohort; `sep` controls separability, `flip` label noise."""
    X0 = rng.randn(n // 2, 20)
    X1 = rng.randn(n // 2, 20) + sep
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    if flip:
        idx = rng.choice(n, size=int(n * flip), replace=False)
        y[idx] = 1 - y[idx]
    return X, y


# ── check_composition ─────────────────────────────────────────────────────────

def test_composition_detects_class_imbalance():
    y_s = np.array([0] * 60 + [1] * 40)
    y_t = np.array([0] * 30 + [1] * 70)
    r = check_composition(y_s, y_t)
    assert r["class_differs"] is True
    assert r["class_p"] < 0.05
    assert any("balanced_accuracy" in a for a in r["advice"])


def test_composition_same_proportions():
    y_s = np.array([0] * 50 + [1] * 50)
    y_t = np.array([0] * 25 + [1] * 25)
    r = check_composition(y_s, y_t)
    assert r["class_differs"] is False
    assert r["subgroup_p"] is None


def test_composition_subtype_check_like_esca():
    # 36% vs 97% squamous, as in the TCGA esophageal comparison
    sub_s = ["squamous"] * 35 + ["adeno"] * 62
    sub_t = ["squamous"] * 38 + ["adeno"] * 1
    r = check_composition([0] * 97, [0] * 39, sub_s, sub_t)
    assert r["subgroup_differs"] is True
    assert r["subgroup_p"] < 1e-6
    assert any("Subtype mix differs" in a for a in r["advice"])


def test_composition_drops_missing_subtypes():
    sub_s = ["a", "b", None, float("nan"), ""] * 10
    sub_t = ["a", "b"] * 10
    r = check_composition([0] * 50, [0] * 20, sub_s, sub_t)
    assert sum(r["subgroup_counts_source"]) == 20


# ── direction-aware recommendation ────────────────────────────────────────────

def test_target_more_accurate_is_not_correction_required():
    rng = np.random.RandomState(1)
    X_s, y_s = _cohort(rng, 400, sep=0.6, flip=0.15)   # hard, noisy source
    X_t, y_t = _cohort(rng, 200, sep=2.5)               # easy target
    fw = AncestryAuditFramework(random_state=0, n_bootstrap=300)
    rep = fw.audit(LogisticRegression(max_iter=1000), X_s, y_s, X_t, y_t)
    assert rep.gap_pp < 0
    assert rep.p_value < 0.05
    assert rep.recommendation == "target_more_accurate"


def test_correction_required_when_target_worse_and_balanced():
    rng = np.random.RandomState(2)
    X_s, y_s = _cohort(rng, 400, sep=2.5)
    X_t, y_t = _cohort(rng, 200, sep=2.5, flip=0.25)    # noisy target labels
    fw = AncestryAuditFramework(random_state=0, n_bootstrap=300)
    rep = fw.audit(LogisticRegression(max_iter=1000), X_s, y_s, X_t, y_t)
    assert rep.gap_pp > 0
    assert rep.recommendation == "correction_required"
    assert rep.notes


def test_check_composition_when_class_mix_differs():
    rng = np.random.RandomState(3)
    X_s, y_s = _cohort(rng, 400, sep=2.5)
    X_t, y_t = _cohort(rng, 400, sep=2.5, flip=0.25)
    keep = np.r_[np.where(y_t == 0)[0][:40], np.where(y_t == 1)[0][:160]]
    X_t, y_t = X_t[keep], y_t[keep]                      # 20/80 class mix
    fw = AncestryAuditFramework(random_state=0, n_bootstrap=300)
    rep = fw.audit(LogisticRegression(max_iter=1000), X_s, y_s, X_t, y_t)
    assert rep.class_composition_p < 0.05
    if rep.gap_pp > fw.threshold_pp and rep.p_value < fw.threshold_p:
        assert rep.recommendation == "check_composition"


def test_no_action_on_null():
    rng = np.random.RandomState(4)
    X_s, y_s = _cohort(rng, 300, sep=1.0)
    X_t, y_t = _cohort(rng, 150, sep=1.0)
    fw = AncestryAuditFramework(random_state=0, n_bootstrap=300)
    rep = fw.audit(LogisticRegression(max_iter=1000), X_s, y_s, X_t, y_t)
    if rep.p_value >= 0.05:
        assert rep.recommendation == "no_action"
        assert any("power_analysis" in n for n in rep.notes)


# ── correct() warning ─────────────────────────────────────────────────────────

def test_correct_warns_without_audit():
    rng = np.random.RandomState(5)
    X_s, y_s = _cohort(rng, 200, sep=1.0)
    X_t, y_t = _cohort(rng, 120, sep=1.0)
    fw = AncestryAuditFramework(random_state=0)
    with pytest.warns(UserWarning, match="before audit"):
        fw.correct(LogisticRegression(max_iter=1000), X_s, y_s, X_t, y_t, n_samples=40)


def test_correct_warns_when_not_recommended():
    rng = np.random.RandomState(6)
    X_s, y_s = _cohort(rng, 400, sep=0.6, flip=0.15)
    X_t, y_t = _cohort(rng, 200, sep=2.5)
    fw = AncestryAuditFramework(random_state=0, n_bootstrap=300)
    fw.audit(LogisticRegression(max_iter=1000), X_s, y_s, X_t, y_t)
    with pytest.warns(UserWarning, match="target_more_accurate"):
        fw.correct(LogisticRegression(max_iter=1000), X_s, y_s, X_t, y_t, n_samples=40)


def test_correct_silent_when_recommended():
    rng = np.random.RandomState(2)
    X_s, y_s = _cohort(rng, 400, sep=2.5)
    X_t, y_t = _cohort(rng, 200, sep=2.5, flip=0.25)
    fw = AncestryAuditFramework(random_state=0, n_bootstrap=300)
    rep = fw.audit(LogisticRegression(max_iter=1000), X_s, y_s, X_t, y_t)
    assert rep.recommendation == "correction_required"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fw.correct(LogisticRegression(max_iter=1000), X_s, y_s, X_t, y_t, n_samples=40)


# ── audit_many ────────────────────────────────────────────────────────────────

def test_audit_many_holm_and_verdict(tmp_path):
    rng = np.random.RandomState(7)
    X_s, y_s = _cohort(rng, 400, sep=0.6, flip=0.15)
    X_t, y_t = _cohort(rng, 200, sep=2.5)
    models = {
        "LR1": LogisticRegression(max_iter=1000, C=1.0),
        "LR2": LogisticRegression(max_iter=1000, C=0.1),
    }
    fw = AncestryAuditFramework(random_state=0, n_bootstrap=300)
    mr = fw.audit_many(models, X_s, y_s, X_t, y_t, metric="balanced_accuracy")
    assert mr.n_models == 2
    assert all(mr.p_holm[k] >= mr.p_raw[k] - 1e-12 for k in models)
    assert mr.verdict == "target_more_accurate"
    fw.diagnose(y_s, y_t)
    report = fw.generate_report(save_path=str(tmp_path / "r.json"))
    assert "audit_many" in report and "diagnose" in report
    assert "AUDIT x2" in fw.summary()
