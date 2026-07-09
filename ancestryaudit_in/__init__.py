"""
AncestryAudit v0.1.0
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

__version__ = "0.1.0"
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
    ci_95:           Tuple[float, float]
    source_accuracy: float
    target_accuracy: float
    n_source:        int
    n_target:        int
    recommendation:  str     # "correction_required" | "no_action"
    # Private fields — excluded from repr
    _model:      Any        = field(default=None, repr=False)
    _boot_gaps:  List[float] = field(default_factory=list, repr=False)

    def __str__(self) -> str:
        return (
            f"AuditReport(gap={self.gap_pp:+.2f}pp, "
            f"p={self.p_value:.4f}, d={self.cohen_d:.3f}, "
            f"95%CI=[{self.ci_95[0]:.2f}, {self.ci_95[1]:.2f}], "
            f"recommendation='{self.recommendation}')"
        )


@dataclass
class CorrectionReport:
    """Results from AncestryAuditFramework.correct()."""
    delta_pp:          float
    p_value:           float
    n_used:            int
    seed_robustness:   Dict[str, Any]
    baseline_accuracy: float
    corrected_accuracy: float
    all_positive:      bool

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
        Bootstrap iterations for CI and p-value estimation.
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
        n_bootstrap:  int = 1000,
        threshold_pp: float = 2.0,
        threshold_p:  float = 0.05,
    ):
        self.random_state = random_state
        self.n_bootstrap  = n_bootstrap
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
        Bootstraps the gap for CI and p-value.

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
            p_value : two-sided p-value from bootstrap t-test
            cohen_d : effect size
            ci_95 : 95% bootstrap CI on gap_pp
            recommendation : 'correction_required' if |gap|>threshold_pp
                             AND p<threshold_p, else 'no_action'
        """
        results = compute_audit(
            model, X_source, y_source, X_target, y_target,
            n_bootstrap=self.n_bootstrap,
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
            ci_95=results["ci_95"],
            source_accuracy=results["source_accuracy"],
            target_accuracy=results["target_accuracy"],
            n_source=results["n_source"],
            n_target=results["n_target"],
            recommendation=recommendation,
            _model=results["trained_model"],
            _boot_gaps=results["boot_gaps"],
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
        retrains the model, and validates robustness across 10 random seeds.

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
            delta_pp : mean accuracy improvement on target holdout (pp)
            p_value : two-sided t-test on per-seed deltas vs 0
            n_used : actual samples used (min(n_samples, available))
            seed_robustness : mean, sd, min, max, n_positive across 10 seeds
            all_positive : True if all 10 seeds showed positive correction
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
            seed_robustness=results["seed_robustness"],
            baseline_accuracy=results["baseline_accuracy"],
            corrected_accuracy=results["corrected_accuracy"],
            all_positive=results["all_positive"],
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
                f"           95% CI [{ar.ci_95[0]:.2f}, {ar.ci_95[1]:.2f}]pp",
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
