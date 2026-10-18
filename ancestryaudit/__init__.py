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
import warnings

import numpy as np

from .audit      import compute_audit, run_per_algorithm_audit, holm_bonferroni
from .diagnose   import check_composition
from .correction import apply_correction
from .filter     import filter_noise
from .report     import generate_full_report

__version__ = "0.4.0"
__author__  = "Dana Yergaliyeva"
__all__     = [
    "AncestryAuditFramework",
    "AuditReport",
    "CorrectionReport",
    "ValidationReport",
    "CompositionReport",
    "MultiAuditReport",
    "check_composition",
    "run_per_algorithm_audit",
    "holm_bonferroni",
]

# Possible values of AuditReport.recommendation
RECOMMENDATIONS = (
    "no_action",             # no significant gap above threshold_pp
    "target_more_accurate",  # significant gap in favor of the target group
    "check_composition",     # significant gap, but class proportions differ
                             # and metric="accuracy": rerun with balanced accuracy
    "correction_required",   # significant gap against the target group
)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class AuditReport:
    """Results from AncestryAuditFramework.audit()."""
    gap_pp:          float   # positive = source better; negative = target better
    p_value:         float
    cohen_d:         Optional[float]   # None when metric="balanced_accuracy"
    null_ci:         Tuple[float, float]   # 2.5/97.5 percentiles of permutation null distribution (NOT a CI on the gap)
    source_accuracy: float
    target_accuracy: float
    n_source:        int
    n_target:        int
    recommendation:  str     # one of RECOMMENDATIONS
    metric:          str = "accuracy"  # "accuracy" or "balanced_accuracy"
    class_composition_p: Optional[float] = None  # chi-square p for class proportions
    notes:           List[str] = field(default_factory=list)  # plain-language advice
    # Private fields — excluded from repr
    _model:      Any        = field(default=None, repr=False)

    def __str__(self) -> str:
        d_str = f"{self.cohen_d:.3f}" if self.cohen_d is not None else "N/A"
        return (
            f"AuditReport(gap={self.gap_pp:+.2f}pp, "
            f"p={self.p_value:.4f}, d={d_str}, "
            f"metric={self.metric}, "
            f"null_dist_spread=[{self.null_ci[0]:.2f}, {self.null_ci[1]:.2f}], "
            f"recommendation='{self.recommendation}')"
        )


@dataclass
class CompositionReport:
    """Results from AncestryAuditFramework.diagnose()."""
    class_props_source: List[float]
    class_props_target: List[float]
    class_p:            Optional[float]
    class_differs:      bool
    subgroup_labels:    Optional[List[Any]] = None
    subgroup_props_source: Optional[List[float]] = None
    subgroup_props_target: Optional[List[float]] = None
    subgroup_p:         Optional[float] = None
    subgroup_differs:   Optional[bool] = None
    advice:             List[str] = field(default_factory=list)
    details:            Dict[str, Any] = field(default_factory=dict, repr=False)

    def __str__(self) -> str:
        cp = f"{self.class_p:.3g}" if self.class_p is not None else "N/A"
        sp = f"{self.subgroup_p:.3g}" if self.subgroup_p is not None else "not checked"
        return (
            f"CompositionReport(class_p={cp}, class_differs={self.class_differs}, "
            f"subgroup_p={sp}, subgroup_differs={self.subgroup_differs})"
        )


@dataclass
class MultiAuditReport:
    """Results from AncestryAuditFramework.audit_many()."""
    metric:         str
    n_models:       int
    n_significant:  int          # after Holm correction
    verdict:        str          # see audit_many() docstring
    gaps_pp:        Dict[str, float]
    p_raw:          Dict[str, float]
    p_holm:         Dict[str, float]
    significant:    Dict[str, bool]

    def __str__(self) -> str:
        return (
            f"MultiAuditReport(metric={self.metric}, "
            f"significant={self.n_significant}/{self.n_models} (Holm), "
            f"verdict='{self.verdict}')"
        )


@dataclass
class CorrectionReport:
    """Results from AncestryAuditFramework.correct()."""
    delta_pp:             float
    p_value:              float   # McNemar exact/chi-square p-value
    n_used:               int
    refit_robustness:     Dict[str, Any]
    baseline_accuracy:    float
    corrected_accuracy:   float   # leaky/optimistic - see corrected_accuracy_holdout
    all_positive:         bool
    direction_confirmed:  bool = False  # True if McNemar p<0.05 and fine-tuned model beat baseline (c>b)
    corrected_accuracy_holdout: Optional[float] = None  # clean, scored only on hold_primary
    per_class_holdout:    Dict[str, Any] = field(default_factory=dict)  # per-class baseline/corrected accuracy on holdout - inspect for asymmetric (majority-class-driven) correction

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
        Minimum |gap| in percentage points for a gap to count as
        actionable (either direction).
    threshold_p : float, default=0.05
        Maximum p-value for a gap to count as significant.

    Recommended order: diagnose() -> audit() / audit_many() ->
    correct() only if the recommendation is 'correction_required' ->
    validate().

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
        self._composition_report: Optional[CompositionReport] = None
        self._multi_audit_report: Optional[MultiAuditReport]  = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def audit(self, model, X_source, y_source,
              X_target, y_target, metric: str = "accuracy") -> AuditReport:
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
        metric : {"accuracy", "balanced_accuracy"}, default "accuracy"
            "balanced_accuracy" is robust to differing class priors
            between source and target - use this whenever the two
            groups may have different class balance, since a raw
            accuracy gap can otherwise reflect prior mismatch rather
            than genuine ancestry-linked signal. cohen_d is None when
            metric="balanced_accuracy".

        Returns
        -------
        AuditReport
            gap_pp : accuracy gap in percentage points (positive = source better)
            p_value : two-sided p-value from label-permutation test
            cohen_d : effect size (accuracy metric only; None for balanced_accuracy)
            null_ci : 2.5/97.5 percentiles of the permutation null distribution
            (centered near zero — NOT a CI on the observed gap)
            recommendation : one of
                'no_action'            - no significant gap above threshold_pp
                'target_more_accurate' - significant gap in favor of the
                                         target group; no correction needed
                'check_composition'    - significant gap against the target,
                                         but class proportions differ and
                                         metric="accuracy": rerun with
                                         metric="balanced_accuracy" first
                'correction_required'  - significant gap against the target
                                         that class proportions do not
                                         explain. Check subtypes with
                                         diagnose() before correcting.
            class_composition_p : chi-square p-value for class proportions
            notes : plain-language advice
        """
        results = compute_audit(
            model, X_source, y_source, X_target, y_target,
            n_permutations=self.n_bootstrap,
            random_state=self.random_state,
            metric=metric,
        )

        gap_pp  = results["gap_pp"]
        p_value = results["p_value"]

        comp = check_composition(y_source, y_target)
        class_p = comp["class_p"]
        significant = (abs(gap_pp) > self.threshold_pp
                       and p_value < self.threshold_p)
        notes: List[str] = []

        if not significant:
            recommendation = "no_action"
            notes.append(
                "No significant gap. This is not proof of fairness: run "
                "power_analysis() to see how large a gap these sample sizes "
                "can detect.")
        elif gap_pp < 0:
            recommendation = "target_more_accurate"
            notes.append(
                "The model is significantly MORE accurate for the target "
                "group. No correction is needed; check composition with "
                "diagnose() to explain the difference.")
        elif metric == "accuracy" and comp["class_differs"]:
            recommendation = "check_composition"
            notes.append(
                "Class proportions differ between cohorts "
                f"(p={class_p:.2g}), so raw accuracy is confounded by class "
                "mix. Rerun audit() with metric='balanced_accuracy' before "
                "any correction.")
        else:
            recommendation = "correction_required"
            notes.append(
                "Significant gap against the target group. Before correcting, "
                "check subtype composition with diagnose(); if several models "
                "were audited, use audit_many() for Holm correction.")

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
            metric=results.get("metric", metric),
            class_composition_p=class_p,
            notes=notes,
            _model=results["trained_model"],
        )
        return self._audit_report

    def diagnose(self, y_source, y_target,
                 subgroup_source=None, subgroup_target=None,
                 alpha: float = 0.05) -> CompositionReport:
        """
        Check whether the two cohorts differ in composition.

        Compares class proportions (chi-square) and, if subtype labels are
        given, the subtype mix within a class of interest. Run this before
        attributing any accuracy gap to ancestry.

        Parameters
        ----------
        y_source, y_target : array-like — class labels of each cohort
        subgroup_source, subgroup_target : sequence, optional — subtype
            label per sample (e.g. "squamous" / "adenocarcinoma" for the
            esophageal tumors of each cohort). Missing values are dropped.
        alpha : float, default 0.05

        Returns
        -------
        CompositionReport (full results in .details, advice in .advice)
        """
        r = check_composition(y_source, y_target,
                              subgroup_source, subgroup_target, alpha=alpha)
        self._composition_report = CompositionReport(
            class_props_source=r["class_props_source"],
            class_props_target=r["class_props_target"],
            class_p=r["class_p"],
            class_differs=r["class_differs"],
            subgroup_labels=r["subgroup_labels"],
            subgroup_props_source=r["subgroup_props_source"],
            subgroup_props_target=r["subgroup_props_target"],
            subgroup_p=r["subgroup_p"],
            subgroup_differs=r["subgroup_differs"],
            advice=r["advice"],
            details=r,
        )
        return self._composition_report

    def audit_many(self, models: Dict[str, Any], X_source, y_source,
                   X_target, y_target,
                   metric: str = "balanced_accuracy") -> MultiAuditReport:
        """
        Audit several models at once with Holm correction.

        Testing several models at the 5% level inflates the chance that at
        least one looks significant by chance (about 26% for six models).
        This runs the permutation test for each model and adjusts the
        p-values with the Holm step-down procedure.

        Parameters
        ----------
        models : dict name -> unfitted sklearn-compatible estimator
        X_source, y_source : FULL source cohort (split internally 75/25)
        X_target, y_target : target cohort
        metric : {"accuracy", "balanced_accuracy"}, default "balanced_accuracy"

        Returns
        -------
        MultiAuditReport with verdict:
            'no_significant_gap'     - no model significant after Holm
            'target_less_accurate'   - all significant models favor the source
            'target_more_accurate'   - all significant models favor the target
            'inconsistent_direction' - significant models disagree in direction
        """
        results, n_sig = run_per_algorithm_audit(
            models, X_source, y_source, X_target, y_target,
            metric=metric, n_permutations=self.n_bootstrap,
            random_state=self.random_state, alpha=self.threshold_p,
        )
        sig_gaps = [r["gap_pp"] for r in results.values() if r["significant_holm"]]
        if not sig_gaps:
            verdict = "no_significant_gap"
        elif all(g > 0 for g in sig_gaps):
            verdict = "target_less_accurate"
        elif all(g < 0 for g in sig_gaps):
            verdict = "target_more_accurate"
        else:
            verdict = "inconsistent_direction"

        self._multi_audit_report = MultiAuditReport(
            metric=metric,
            n_models=len(results),
            n_significant=int(n_sig),
            verdict=verdict,
            gaps_pp={k: float(r["gap_pp"]) for k, r in results.items()},
            p_raw={k: float(r["p_value"]) for k, r in results.items()},
            p_holm={k: float(r["p_holm"]) for k, r in results.items()},
            significant={k: bool(r["significant_holm"]) for k, r in results.items()},
        )
        return self._multi_audit_report

    def filter_stratification_noise(
        self,
        X,
        gene_list: List[str],
        gene_biotype: Optional[Dict[str, str]] = None,
    ):
        """
        Remove ancestry-linked CNV regions unrelated to cancer biology.

        Removes olfactory receptors (OR*), pseudogenes, and
        uncharacterized clone-based loci (names containing '.').

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_genes)
        gene_list : list of str — gene names for each column
        gene_biotype : dict, optional — Ensembl/GENCODE biotype per gene
            symbol. STRONGLY RECOMMENDED: without this, pseudogene removal
            falls back to an unreliable name-pattern heuristic (raises
            UserWarning). Pass this whenever you have it.

        Returns
        -------
        X_filtered : same type as X, with excluded columns removed
        kept_genes : list of str
        filter_log : dict with removal statistics and Methods disclosure text
        """
        X_filtered, kept_genes, filter_log = filter_noise(
            X, gene_list, gene_biotype=gene_biotype)
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

        Note: corrected_accuracy is scored on the full X_target (includes
        fine-tuning data, leaky/optimistic). Use corrected_accuracy_holdout
        for a fair comparison against baseline_accuracy.
        """
        if self._audit_report is None:
            warnings.warn(
                "correct() was called before audit(): no gap has been "
                "confirmed. Run audit() (and diagnose()) first; fine-tuning "
                "with target samples can raise accuracy even when there is "
                "no bias, so the result is not evidence of a corrected bias.",
                UserWarning, stacklevel=2)
        elif self._audit_report.recommendation != "correction_required":
            warnings.warn(
                "correct() was called although the last audit recommended "
                f"'{self._audit_report.recommendation}'. The accuracy change "
                "from fine-tuning should not be reported as a corrected bias.",
                UserWarning, stacklevel=2)

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
            corrected_accuracy_holdout=results.get("corrected_accuracy_holdout"),
            all_positive=results.get("all_positive", False),
            direction_confirmed=results.get("direction_confirmed", False),
            per_class_holdout=results.get("per_class_holdout", {}),
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
            pre_gap : performance gap before correction (pp), base model
                scored on X_holdout (NOT the original audit's target_accuracy,
                which was a different sample set - re-scoring on the same
                holdout as post_gap makes this a fair before/after comparison)
            post_gap : performance gap after correction (pp)
            correction_magnitude : pre_gap - post_gap
            improvement_pp : accuracy improvement on target
        """
        if self._audit_report is None:
            raise RuntimeError(
                "Run audit() before validate(). "
                "validate() needs the base model from the audit step."
            )
        if self._audit_report._model is None:
            raise RuntimeError(
                "audit_report._model is missing - cannot re-score the base "
                "model on X_holdout for a fair before/after comparison."
            )

        X_h = _to_numpy(X_holdout)
        y_h = np.array(y_holdout)

        from sklearn.metrics import accuracy_score as _accuracy_score
        base_model = self._audit_report._model
        pre_acc  = float(_accuracy_score(y_h, base_model.predict(X_h)))
        post_acc = float(_accuracy_score(y_h, corrected_model.predict(X_h)))
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

        Power here is the chance of detecting a gap AGAINST the target
        group (significant and gap_pp > 0): the permutation test itself is
        two-sided, but only a target-worse gap counts as a detection, since
        that is the case a correction would address.

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
        dict with power_pct, recommendation, n_target_needed, n_source_needed,
             flip_rate_used, expected_gap_pp, n_source, n_target
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
        n_needed_confirmed = True
        if power_pct < 80:
            lo, hi = n_target, 5000
            for _ in range(8):
                mid = (lo + hi) // 2
                p = estimate_power(mid * 2, mid, seed_offset=999)
                if p >= 80:
                    hi = mid
                else:
                    lo = mid
            # Bisection is capped at hi=5000. If the search never actually
            # observed >=80% power at any tested n (i.e. every midpoint
            # failed), hi is just the unverified search ceiling, not a
            # confirmed answer - a prior version silently returned it as
            # if it were. Verify explicitly before reporting it as SUFFICIENT.
            p_at_hi = estimate_power(hi * 2, hi, seed_offset=999)
            n_needed = hi
            n_needed_confirmed = bool(p_at_hi >= 80)
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
            if n_needed_confirmed:
                print(f"  n_target for 80% pwr : ~{n_needed:,}")
                print(f"  n_source for 80% pwr : ~{n_needed*2:,}")
            else:
                print(f"  n_target for 80% pwr : >{n_needed:,} (search ceiling "
                      f"reached - even {n_needed:,} target samples did not "
                      f"reach 80% power at this expected_gap_pp; true "
                      f"n_needed exceeds the search range and is unconfirmed)")
            print(f"  (both sides must scale together)")
        print("  CAVEAT: power/MDE estimated on synthetic i.i.d. Gaussian")
        print("  features, not real correlated CNV data. Treat as a rough")
        print("  benchmark, not an empirical sample-size guarantee.")
        print("=" * 58)

        return {
            "power_pct":          power_pct,
            "recommendation":     recommendation,
            "n_target_needed":    n_needed,
            "n_source_needed":    n_needed * 2 if n_needed else None,
            "n_needed_confirmed": n_needed_confirmed,
            "flip_rate_used":     flip_rate,
            "expected_gap_pp":    expected_gap_pp,
            "n_source":           n_source,
            "n_target":           n_target,
            "caveat": ("Estimated on synthetic i.i.d. Gaussian features, "
                       "not real correlated CNV data - a rough benchmark, "
                       "not an empirical guarantee."),
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
            composition_report=self._composition_report,
            multi_audit_report=self._multi_audit_report,
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

        if self._composition_report:
            cr0 = self._composition_report
            cp = f"{cr0.class_p:.3g}" if cr0.class_p is not None else "N/A"
            sp = (f"{cr0.subgroup_p:.3g}" if cr0.subgroup_p is not None
                  else "not checked")
            lines += [
                f"\n  [DIAGNOSE] Class mix p = {cp}  (differs: {cr0.class_differs})",
                f"             Subtype mix p = {sp}",
            ] + [f"             - {a}" for a in cr0.advice]

        if self._multi_audit_report:
            mr = self._multi_audit_report
            lines += [
                f"\n  [AUDIT x{mr.n_models}] {mr.n_significant}/{mr.n_models} significant "
                f"after Holm (metric={mr.metric})",
                f"             → {mr.verdict}",
            ]

        if self._audit_report:
            ar = self._audit_report
            d_str = f"{ar.cohen_d:.3f}" if ar.cohen_d is not None else "N/A"
            lines += [
                f"\n  [AUDIT]  Gap = {ar.gap_pp:+.2f}pp  (metric={ar.metric})",
                f"           p = {ar.p_value:.4f}  d = {d_str}",
                f"           Null CI [{ar.null_ci[0]:.2f}, {ar.null_ci[1]:.2f}]pp (permutation null spread)",
                f"           → {ar.recommendation}",
            ] + [f"           - {n}" for n in ar.notes]

        if self._correction_report:
            cr = self._correction_report
            lines += [
                f"\n  [CORRECT] Δ = {cr.delta_pp:+.2f}pp  p = {cr.p_value:.4f}",
                f"            n_used = {cr.n_used}",
                f"            All 5 refits positive: {cr.all_positive}",
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
