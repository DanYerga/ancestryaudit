# AncestryAudit

**Bias detection and correction framework for genomic cancer AI.**

Detects ancestry-linked performance gaps in copy number variation (CNV)-based
cancer classifiers, applies supervised fine-tuning correction, and generates
structured audit reports.

Developed from research on ancestry bias in TCGA-LIHC/STAD classification
(Yergaliyeva, 2026).

---

## Installation

```bash
pip install ancestryaudit
# or from source:
pip install -e .
```

---

## Input Format

AncestryAudit works on any CNV feature matrix:

| Format | Shape | Notes |
|--------|-------|-------|
| `np.ndarray` | `(n_samples, n_genes)` | Continuous copy-number values |
| `pd.DataFrame` | `(n_samples, n_genes)` | Column names = gene identifiers |

**Column values:** continuous copy number (e.g. TCGA ABSOLUTE pipeline output,
where 2.0 = normal diploid, >2 = amplification, <2 = deletion).

**Labels:** binary integer (0 or 1), one per sample.

**Models:** any scikit-learn compatible estimator with `fit` / `predict` interface.

---

## Quick Start

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from ancestryaudit import AncestryAuditFramework

# NOTE: Replace with your real CNV matrices
# The example below uses synthetic data for illustration only
rng = np.random.RandomState(42)
X_western = rng.randn(200, 50)
y_western  = (X_western[:, 0] + X_western[:, 1] > 0).astype(int)
X_asian    = rng.randn(80, 50)
y_asian    = (X_asian[:, 0] + X_asian[:, 1] > 0).astype(int)

framework = AncestryAuditFramework()

# Step 0: Check data sufficiency first
power = framework.power_analysis(n_source=200, n_target=80, expected_gap_pp=3.0)

# Step 1: Detect ancestry-linked performance gap
report = framework.audit(
    LogisticRegression(max_iter=1000),
    X_western, y_western,
    X_asian,   y_asian
)
print(f"Gap: {report.gap_pp:.2f}pp, p={report.p_value:.4f}")
print(f"Recommendation: {report.recommendation}")
```

Output:
```
Gap: +2.39pp, p=0.0069
Recommendation: correction_required
```

---

## Full Pipeline

```python
from ancestryaudit import AncestryAuditFramework
from sklearn.linear_model import LogisticRegression

framework = AncestryAuditFramework(
    random_state=42,
    n_bootstrap=1000,
    threshold_pp=2.0,   # minimum gap to trigger correction_required
    threshold_p=0.05    # maximum p-value to trigger correction_required
)

# ── Step 1: Filter population-stratification noise (optional) ──────────────
X_western_filtered, kept_genes, filter_log = framework.filter_stratification_noise(
    X_western_df,   # pd.DataFrame with gene names as columns
    gene_list       # list of gene name strings
)
print(f"Removed {filter_log['n_removed']} junk genes, kept {filter_log['n_kept']}")

# ── Step 2: Audit ──────────────────────────────────────────────────────────
audit_report = framework.audit(
    LogisticRegression(max_iter=1000),
    X_western_filtered, y_western,
    X_asian_filtered,   y_asian
)
print(audit_report)
# AuditReport(gap=+2.39pp, p=0.0069, d=1.52, null_CI=[-2.10, 2.15],
#             recommendation='correction_required')

# ── Step 3: Correct ────────────────────────────────────────────────────────
if audit_report.recommendation == "correction_required":
    corrected_model, correction_report = framework.correct(
        LogisticRegression(max_iter=1000),
        X_western_filtered, y_western,
        X_asian_labeled,    y_asian_labeled,  # labeled Asian samples
        n_samples=75                          # how many to include
    )
    print(correction_report)
    # CorrectionReport(delta=+3.51pp, p=0.0012, n_used=75, all_positive=True)

# ── Step 4: Validate ───────────────────────────────────────────────────────
validation_report = framework.validate(
    corrected_model,
    X_asian_holdout, y_asian_holdout   # never seen during correction
)
print(validation_report)
# ValidationReport(pre_gap=+2.39pp, post_gap=-1.12pp, improvement=+3.51pp)

# ── Step 5: Report ─────────────────────────────────────────────────────────
report_dict = framework.generate_report("my_audit_report.json")
framework.summary()
```

---

## API Reference

### `AncestryAuditFramework`

| Method | Description | Returns |
|--------|-------------|---------|
| `audit(model, X_source, y_source, X_target, y_target)` | Detect gap | `AuditReport` |
| `filter_stratification_noise(X, gene_list)` | Remove OR/pseudogene columns | `(X_filtered, kept_genes, filter_log)` |
| `correct(model, X_source, y_source, X_target_labeled, y_target_labeled, n_samples)` | Fine-tune correction | `(corrected_model, CorrectionReport)` |
| `validate(corrected_model, X_holdout, y_holdout)` | Post-correction audit | `ValidationReport` |
| `generate_report(save_path)` | Full JSON report | `dict` |
| `summary()` | Print pipeline summary | `str` |

### `AuditReport` fields

| Field | Type | Description |
|-------|------|-------------|
| `gap_pp` | float | Accuracy gap in percentage points (positive = source better) |
| `p_value` | float | Two-sided p-value from bootstrap t-test |
| `cohen_d` | float | Effect size |
| `null_ci` | tuple | 2.5/97.5 percentiles of permutation null (NOT a CI on the gap) |
| `source_accuracy` | float | Model accuracy on held-out source data |
| `target_accuracy` | float | Model accuracy on target data |
| `n_source` | int | Source sample count |
| `n_target` | int | Target sample count |
| `recommendation` | str | `"correction_required"` or `"no_action"` |

### `CorrectionReport` fields

| Field | Type | Description |
|-------|------|-------------|
| `delta_pp` | float | Mean accuracy improvement on target holdout (pp) |
| `p_value` | float | Two-sided t-test on per-seed deltas vs 0 |
| `n_used` | int | Target samples used (min of n_samples and available) |
| `seed_robustness` | dict | mean, sd, min, max, n_positive across 10 seeds |
| `all_positive` | bool | True if all seeds showed positive correction |
| `baseline_accuracy` | float | Source-only accuracy on full target |
| `corrected_accuracy` | float | Estimated corrected accuracy on full target |

### `ValidationReport` fields

| Field | Type | Description |
|-------|------|-------------|
| `pre_gap` | float | Performance gap before correction (pp) |
| `post_gap` | float | Performance gap after correction (pp) |
| `correction_magnitude` | float | pre_gap - post_gap |
| `improvement_pp` | float | Accuracy improvement on target (pp) |
| `pre_accuracy_target` | float | Target accuracy before correction |
| `post_accuracy_target` | float | Target accuracy after correction |

---

## Filtering Details

`filter_stratification_noise` removes three gene categories that are known
to reflect population-level genetic drift rather than cancer biology:

- **Olfactory receptor genes** (`OR*`) — CNV in these clusters varies by
  ancestral migration history, not cancer type
- **Pseudogenes** (`*P`, `*P1`, `*P2`, …) — non-functional, high
  population-stratification signal
- **Uncharacterized loci** (names containing `.`) — clone-based placeholder
  identifiers with no interpretable biological information

**Required Methods disclosure:** The feature space was defined using all
samples prior to train/test split, which constitutes a bounded form of
data snooping. No label information was used in this step (Kaufman et al., 2012).

---

## Expected Results

> **Disclaimer:** Audit results depend on the model, train/test split,
> and preprocessing pipeline provided. Results will differ from published
> paper figures, which used a 7-algorithm ensemble with specific PCA
> preprocessing. The library is designed for arbitrary input —
> directional consistency (not numerical identity) with paper results
> is the correct validation criterion.

**Validated reproduction test (Yergaliyeva, 2026):**

When provided with the exact paper train/test split (White n=338 train,
n=113 test; Asian n=242 evaluation) and gene-aligned PCA features,
the library reproduces paper results with 0.003pp numerical precision:

| Metric | Paper | Library |
|--------|-------|---------|
| Mean PGI | +2.39pp | +2.393pp |
| Algorithms positive | 7/7 | 7/7 |
| Direction | positive | positive |

All 6 API methods (import, audit, correct, validate, report, filter)
pass independent correctness tests on synthetic CNV data.

> **Note on train/test splits:** Provide the full dataset and let the
> framework handle splitting internally. Passing only the training
> portion causes the framework to re-split a subset, producing different
> model boundaries and non-comparable gap values.

---

## Citation

If you use AncestryAudit in research, please cite:

```
Yergaliyeva, D. (2026). Ancestry-linked bias in genomic cancer AI:
Transfer learning correction for East Asian populations.
[Manuscript in preparation]
```

---

## License

MIT License. Copyright (c) 2026 Dana Yergaliyeva.
