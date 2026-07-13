"""
AncestryAudit
====================
Bias detection and correction framework for genomic cancer AI.

Detects ancestry-linked performance gaps between source (training)
and target (evaluation) populations, applies supervised fine-tuning
correction, and generates structured reports.

Quick start
-----------
>>> from ancestryaudit import AncestryAuditFramework
>>> from sklearn.linear_model import LogisticRegression
>>>
>>> framework = AncestryAuditFramework()
>>> report = framework.audit(LogisticRegression(), X_western, y_western,
...                          X_asian, y_asian)
>>> print(f"Gap: {report.gap_pp:.2f}pp, p={report.p_value:.4f}")
>>>
>>> if report.recommendation == "correction_required":
...     corrected, crep = framework.correct(
...         LogisticRegression(), X_western, y_western,
...         X_asian_labeled, y_asian_labeled, n_samples=75)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np

from .audit      import compute_audit
from .correction import apply_correction
from .filter     import filter_noise
from .report     import generate_full_report

__version__ = "0.3.10"
__author__  = "Dana Yergaliyeva"
__all__     = [
    "AncestryAuditFramework",
    "AuditReport",
    "CorrectionReport",
    "ValidationReport",
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class AuditReport:
    """Results from AncestryAuditFramework.audit()."""
    gap_pp:          float   # positive = source better; negative = target better
    p_value:         float
    cohen_d:         float
    null_ci:         Tuple[float, float]   # 2.5/97.5 percentiles of permutation null distribution (NOT a CI on the gap)
    source_accuracy: float
    target_accuracy: float
    n_source:        int
    n_target:        int
    recommendation:  str     # "correction_required" | "no_action"
    # Private fields — excluded from repr
    _model:      Any        = field(default=None, repr=False)

    def __str__(self) -> str:
        return (
            f"AuditReport(gap={self.gap_pp:+.2f}pp, "
            f"p={self.p_value:.4f}, d={self.cohen_d:.3f}, "
            f"null_dist_spread=[{self.null_ci[0]:.2f}, {self.null_ci[1]:.2f}], "
            f"recommendation='{self.recommendation}')"
        )


@dataclass
class CorrectionReport:
    """Results from AncestryAuditFramework.correct()."""
    delta_pp:             float
    p_value:              float   # McNemar exact/chi-square p-value
    n_used:               int
    refit_robustness:     Dict[str, Any]
    baseline_accuracy:    float
    corrected_accuracy:   float
    all_positive:         bool
    direction_confirmed:  bool = False  # True if McNemar p<0.05 and fine-tuned model beat baseline (c>b)

    def __str__(self) -> str:
        return (
            f"CorrectionReport(delta={self.delta_pp:+.2f}pp, "
            f"p={self.p_value:.4f}, n_used={self.n_used}, "
            f"all_positive={self.all_positive})"
        )


@dataclass
class ValidationReport:
    """Results from AncestryAuditFramework.validate()."""
    pre_gap:              float
    post_gap:             float
    correction_magnitude: float
    improvement_pp:       float
    pre_accuracy_target:  float
    post_accuracy_target: float

    def __str__(self) -> str:
        return (
            f"ValidationReport(pre_gap={self.pre_gap:+.2f}pp, "
            f"post_gap={self.post_gap:+.2f}pp, "
            f"improvement={self.improvement_pp:+.2f}pp)"
        )


# ── Main class ────────────────────────────────────────────────────────────────

class AncestryAuditFramework:
    """
    Bias detection and correction for genomic cancer AI models.

    Parameters
    ----------
    random_state : int, default=42
        Seed for all random operations.
    n_bootstrap : int, default=1000
        Deprecated alias for n_permutations. Permutation-test iterations
        used to build the null distribution and compute the p-value.
    threshold_pp : float, default=2.0
        Minimum |gap| in percentage points to trigger 'correction_required'.
    threshold_p : float, default=0.05
        Maximum p-value to trigger 'correction_required'.

    Notes
    -----
    Input format: any array-like with shape (n_samples, n_features).
    If using DataFrames, column names are preserved in filter operations.
    Copy number values should be continuous (e.g. TCGA ABSOLUTE pipeline output).
    Models must be sklearn-compatible (fit/predict interface).
    """

    def __init__(
        self,
        random_state: int = 42,
        n_permutations: int = None,
        n_bootstrap: int = None,
        threshold_pp: float = 2.0,
        threshold_p:  float = 0.05,
    ):
        self.random_state = random_state
        if n_bootstrap is not None and n_permutations is None:
            n_permutations = n_bootstrap
        self.n_bootstrap = n_permutations if n_permutations is not None else 1000
        self.threshold_pp = threshold_pp
        self.threshold_p  = threshold_p

        # Internal state — populated as pipeline steps are run
        self._audit_report:      Optional[AuditReport]      = None
        self._correction_report: Optional[CorrectionReport] = None
        self._validation_report: Optional[ValidationReport] = None
        self._filter_log:        Optional[Dict]             = None
        self._corrected_model:   Any                        = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def audit(self, model, X_source, y_source,
              X_target, y_target) -> AuditReport:
        """
        Detect ancestry-linked performance gap.

        Trains model on 75% of source data; evaluates on the held-out 25%
        (source accuracy) and the full target set (target accuracy).
        Runs a label-permutation test for the gap p-value and null distribution.

        Parameters
        ----------
        model : sklearn-compatible estimator (unfitted)
        X_source : array-like, shape (n, p) — source (Western) CNV features
        y_source : array-like — source labels (binary)
        X_target : array-like, shape (m, p) — target (Asian) CNV features
        y_target : array-like — target labels (binary)

        Returns
        -------
        AuditReport
            gap_pp : accuracy gap in percentage points (positive = source better)
            p_value : two-sided p-value from label-permutation test
            cohen_d : effect size
            null_ci : 2.5/97.5 percentiles of the permutation null distribution
            (centered near zero — NOT a CI on the observed gap)
            recommendation : 'correction_required' if |gap|>threshold_pp
                             AND p<threshold_p, else 'no_action'
        """
        results = compute_audit(
            model, X_source, y_source, X_target, y_target,
            n_permutations=self.n_bootstrap,
            random_state=self.random_state,
        )

        gap_pp  = results["gap_pp"]
        p_value = results["p_value"]

        recommendation = (
            "correction_required"
            if abs(gap_pp) > self.threshold_pp and p_value < self.threshold_p
            else "no_action"
        )

        self._audit_report = AuditReport(
            gap_pp=gap_pp,
            p_value=p_value,
            cohen_d=results["cohen_d"],
            null_ci=results.get("null_ci", (0.0, 0.0)),
            source_accuracy=results["source_accuracy"],
            target_accuracy=results["target_accuracy"],
            n_source=results["n_source"],
            n_target=results["n_target"],
            recommendation=recommendation,
            _model=results["trained_model"],
        )
        return self._audit_report

    def filter_stratification_noise(
        self,
        X,
        gene_list: List[str],
    ):
        """
        Remove ancestry-linked CNV regions unrelated to cancer biology.

        Removes olfactory receptors (OR*), pseudogenes (*P), and
        uncharacterized clone-based loci (names containing '.').

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_genes)
        gene_list : list of str — gene names for each column

        Returns
        -------
        X_filtered : same type as X, with excluded columns removed
        kept_genes : list of str
        filter_log : dict with removal statistics and Methods disclosure text
        """
        X_filtered, kept_genes, filter_log = filter_noise(X, gene_list)
        self._filter_log = filter_log
        return X_filtered, kept_genes, filter_log

    def correct(
        self,
        model,
        X_source,
        y_source,
        X_target_labeled,
        y_target_labeled,
        n_samples: int = 75,
    ) -> Tuple[Any, CorrectionReport]:
        """
        Supervised fine-tuning correction.

        Combines source data with n_samples from the labeled target cohort,
        retrains the model. Uses McNemar exact/chi-square test on fixed holdout.
        Robustness: 5 refits varying model random_state, split held fixed.

        Parameters
        ----------
        model : sklearn-compatible estimator
        X_source, y_source : source training data
        X_target_labeled, y_target_labeled : labeled target data
        n_samples : int, target samples to include (stratified by label)

        Returns
        -------
        corrected_model : fitted sklearn estimator
        CorrectionReport
            delta_pp : accuracy improvement on primary fixed-split holdout (pp)
            (single value, not a mean across seeds)
            p_value : McNemar p-value (exact binomial when b+c<25, chi-square otherwise)
            n_used : actual samples used (min(n_samples, available))
            refit_robustness : mean, min, max across 5 refits (split fixed)
            all_positive : True if all 5 refits show positive delta_pp (not just the primary split)
            direction_confirmed : True if McNemar p<0.05 and fine-tuned model beat baseline (c>b)
        """
        corrected_model, results = apply_correction(
            model, X_source, y_source,
            X_target_labeled, y_target_labeled,
            n_samples=n_samples,
            random_state=self.random_state,
        )

        self._corrected_model = corrected_model
        self._correction_report = CorrectionReport(
            delta_pp=results["delta_pp"],
            p_value=results["p_value"],
            n_used=results["n_used"],
            refit_robustness=results.get("refit_robustness", {}),
            baseline_accuracy=results["baseline_accuracy"],
            corrected_accuracy=results["corrected_accuracy"],
            all_positive=results.get("all_positive", False),
            direction_confirmed=results.get("direction_confirmed", False),
        )
        return corrected_model, self._correction_report

    def validate(self, corrected_model, X_holdout, y_holdout) -> ValidationReport:
        """
        Post-correction audit on held-out data.

        Compares pre-correction target accuracy (from audit step) with
        post-correction accuracy on data not used during correction.

        Parameters
        ----------
        corrected_model : fitted corrected estimator
        X_holdout : array-like — target samples held out from correction
        y_holdout : array-like

        Returns
        -------
        ValidationReport
            pre_gap : performance gap before correction (pp)
            post_gap : performance gap after correction (pp)
            correction_magnitude : pre_gap - post_gap
            improvement_pp : accuracy improvement on target
        """
        if self._audit_report is None:
            raise RuntimeError(
                "Run audit() before validate(). "
                "validate() needs the pre-correction target accuracy."
            )

        X_h = _to_numpy(X_holdout)
        y_h = np.array(y_holdout)

        pre_acc  = self._audit_report.target_accuracy
        post_acc = float(
            __import__("sklearn.metrics", fromlist=["accuracy_score"])
            .accuracy_score(y_h, corrected_model.predict(X_h))
        )
        source_acc = self._audit_report.source_accuracy

        pre_gap  = (source_acc - pre_acc)  * 100.0
        post_gap = (source_acc - post_acc) * 100.0

        self._validation_report = ValidationReport(
            pre_gap=pre_gap,
            post_gap=post_gap,
            correction_magnitude=pre_gap - post_gap,
            improvement_pp=(post_acc - pre_acc) * 100.0,
            pre_accuracy_target=pre_acc,
            post_accuracy_target=post_acc,
        )
        return self._validation_report


    def power_analysis(self, n_source, n_target,
                       expected_gap_pp=3.0,
                       n_simulations=200,
                       n_permutations=500):
        """
        Estimate statistical power before running audit().
        Run this FIRST to check whether sample sizes are sufficient.

        Uses empirically calibrated flip_rate.
        n_needed derived from bisection simulation.

        Parameters
        ----------
        n_source : int
        n_target : int
        expected_gap_pp : float
            Expected true gap in percentage points.
            Reference: Saldanha et al. (2024, Nature Medicine)
            found 3-16pp for imaging; use 1-5pp for CNV data.
        n_simulations : int
        n_permutations : int

        Returns
        -------
        dict with power_pct, recommendation,
             n_target_needed, n_source_needed
        """
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from scipy.interpolate import interp1d
        from ancestryaudit.audit import compute_audit

        # Empirical calibration table
        # (n_source=451, n_target=242, n_trials=50)
        CALIB_FLIP = np.array([0.00, 0.01, 0.02, 0.03, 0.05,
                                0.07, 0.10, 0.13, 0.17, 0.20])
        CALIB_GAP  = np.array([-0.256, 0.641, 1.666, 2.410, 4.195,
                                 5.815, 8.294, 10.691, 14.228, 16.608])

        pos_mask    = CALIB_GAP > 0
        gap_to_flip = interp1d(
            CALIB_GAP[pos_mask], CALIB_FLIP[pos_mask],
            kind="linear", fill_value="extrapolate"
        )

        gap_clipped = float(np.clip(
            expected_gap_pp,
            float(CALIB_GAP[pos_mask].min()),
            float(CALIB_GAP[pos_mask].max())
        ))
        flip_rate = float(max(0.0, min(gap_to_flip(gap_clipped), 0.45)))

        model = LogisticRegression(
            max_iter=500, random_state=self.random_state)

        def estimate_power(n_src, n_tgt, seed_offset=0):
            rng = np.random.RandomState(
                self.random_state + seed_offset)
            det = 0
            for trial in range(n_simulations):
                X_s = rng.randn(n_src, 50)
                y_s = (X_s[:,0] + X_s[:,1] > 0).astype(int)
                X_t = rng.randn(n_tgt, 50)
                y_t = (X_t[:,0] + X_t[:,1] > 0).astype(int)
                if flip_rate > 0:
                    n_flip = int(round(n_tgt * flip_rate))
                    if n_flip > 0:
                        idx = rng.choice(
                            n_tgt, size=n_flip, replace=False)
                        y_t[idx] = 1 - y_t[idx]
                r = compute_audit(
                    model, X_s, y_s, X_t, y_t,
                    n_permutations=n_permutations,
                    random_state=trial)
                if (r["p_value"] < self.threshold_p
                        and r["gap_pp"] > 0):
                    det += 1
            return round(det / n_simulations * 100, 1)

        power_pct = estimate_power(n_source, n_target)

        # Bisection for n_needed (both sides scale 2:1)
        if power_pct < 80:
            lo, hi = n_target, 5000
            for _ in range(8):
                mid = (lo + hi) // 2
                p = estimate_power(mid * 2, mid, seed_offset=999)
                if p >= 80:
                    hi = mid
                else:
                    lo = mid
            n_needed = hi
        else:
            n_needed = n_target

        recommendation = (
            "SUFFICIENT" if power_pct >= 80 else "UNDERPOWERED"
        )

        print("=" * 58)
        print("  POWER ANALYSIS  (empirically calibrated)")
        print("=" * 58)
        print(f"  n_source             : {n_source}")
        print(f"  n_target             : {n_target}")
        print(f"  Expected gap         : {expected_gap_pp:.1f}pp")
        print(f"  Calibrated flip_rate : {flip_rate:.4f}")
        print(f"  Estimated power      : {power_pct:.1f}%")
        print(f"  Recommendation       : {recommendation}")
        if recommendation == "UNDERPOWERED":
            print(f"  n_target for 80% pwr : ~{n_needed:,}")
            print(f"  n_source for 80% pwr : ~{n_needed*2:,}")
            print(f"  (both sides must scale together)")
        print("=" * 58)

        return {
            "power_pct":       power_pct,
            "recommendation":  recommendation,
            "n_target_needed": n_needed,
            "n_source_needed": n_needed * 2 if n_needed else None,
            "flip_rate_used":  flip_rate,
            "expected_gap_pp": expected_gap_pp,
            "n_source":        n_source,
            "n_target":        n_target,
        }

    def generate_report(
        self,
        save_path: str = "ancestryaudit_report.json",
    ) -> Dict:
        """
        Generate full JSON report of all completed pipeline steps.

        Parameters
        ----------
        save_path : str, output path (default: ancestryaudit_report.json)

        Returns
        -------
        dict : complete structured report
        """
        return generate_full_report(
            audit_report=self._audit_report,
            correction_report=self._correction_report,
            validation_report=self._validation_report,
            filter_log=self._filter_log,
            save_path=save_path,
        )

    def summary(self) -> str:
        """Print human-readable pipeline summary."""
        lines = ["=" * 55, "  AncestryAudit Pipeline Summary", "=" * 55]

        if self._filter_log:
            fl = self._filter_log
            lines += [
                f"\n  [FILTER] {fl['n_removed']} genes removed "
                f"({fl['removal_pct']:.1f}%)",
                f"           {fl['n_kept']} genes retained",
            ]

        if self._audit_report:
            ar = self._audit_report
            lines += [
                f"\n  [AUDIT]  Gap = {ar.gap_pp:+.2f}pp",
                f"           p = {ar.p_value:.4f}  d = {ar.cohen_d:.3f}",
                f"           Null CI [{ar.null_ci[0]:.2f}, {ar.null_ci[1]:.2f}]pp (permutation null spread)",
                f"           → {ar.recommendation}",
            ]

        if self._correction_report:
            cr = self._correction_report
            lines += [
                f"\n  [CORRECT] Δ = {cr.delta_pp:+.2f}pp  p = {cr.p_value:.4f}",
                f"            n_used = {cr.n_used}",
                f"            Robust across seeds: {cr.all_positive}",
            ]

        if self._validation_report:
            vr = self._validation_report
            lines += [
                f"\n  [VALIDATE] Pre-gap  = {vr.pre_gap:+.2f}pp",
                f"             Post-gap = {vr.post_gap:+.2f}pp",
                f"             Improvement = {vr.improvement_pp:+.2f}pp",
            ]

        lines.append("=" * 55)
        result = "\n".join(lines)
        print(result)
        return result


# ── Module-level helper ────────────────────────────────────────────────────────

def _to_numpy(X):
    if hasattr(X, "values"):
        return X.values.astype(float)
    return np.array(X, dtype=float)
