# AncestryAudit

**Explain an accuracy gap before you correct it.**

AncestryAudit is an open-source Python library for auditing cancer genomic
classifiers across patient populations. It tests whether an accuracy gap
between two groups is statistically significant, estimates the smallest gap
the data can detect, checks whether differences in cancer-type proportions or
subtypes explain the gap, and only then offers a correction.

A lower accuracy in one group is not proof that a model is biased. The gap can
appear by chance, or because the two groups contain a different mix of
cancers. Correcting a model for a bias it does not have can lower its accuracy
while leaving the real cause unaddressed.

---

## Installation

```bash
pip install ancestryaudit
```

Python 3.8 or later. Works with any scikit-learn-compatible model
(`fit` / `predict`).

---

## Quick start

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from ancestryaudit import AncestryAuditFramework

# Synthetic data for illustration: replace with your own feature matrices.
rng = np.random.RandomState(42)
X_source = rng.randn(400, 50)
y_source = (X_source[:, 0] + X_source[:, 1] > 0).astype(int)
X_target = rng.randn(200, 50)
y_target = (X_target[:, 0] + X_target[:, 1] > 0).astype(int)

framework = AncestryAuditFramework(random_state=42)

# 1. Do the cohorts differ in composition?
print(framework.diagnose(y_source, y_target))

# 2. Is there a significant gap?
report = framework.audit(LogisticRegression(max_iter=1000),
                         X_source, y_source, X_target, y_target,
                         metric="balanced_accuracy")
print(report)
for note in report.notes:
    print("-", note)
```

Output:

```
CompositionReport(class_p=0.885, class_differs=False, subgroup_p=not checked, subgroup_differs=None)
AuditReport(gap=-2.13pp, p=0.5790, d=N/A, metric=balanced_accuracy, null_dist_spread=[-6.79, 6.95], recommendation='no_action')
- No significant gap. This is not proof of fairness: run power_analysis() to see how large a gap these sample sizes can detect.
```

---

## Recommended workflow

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

framework = AncestryAuditFramework(random_state=42)

# Step 1: how large a gap can these cohorts detect?
power = framework.power_analysis(n_source=451, n_target=242, expected_gap_pp=3.0)

# Step 2: do the cohorts differ in class proportions or subtypes?
comp = framework.diagnose(y_source, y_target,
                          subgroup_source=subtype_source,   # optional
                          subgroup_target=subtype_target)   # optional
print(comp.advice)

# Step 3: audit several models with Holm correction.
# Wrapping preprocessing in a pipeline fits scaling and PCA on training rows
# only, so no information leaks from the evaluation data.
models = {
    "LR": make_pipeline(StandardScaler(), PCA(50),
                        LogisticRegression(max_iter=1000, class_weight="balanced")),
    "RF": make_pipeline(StandardScaler(), PCA(50),
                        RandomForestClassifier(n_estimators=300, random_state=42)),
}
multi = framework.audit_many(models, X_source, y_source, X_target, y_target,
                             metric="balanced_accuracy")
print(multi.verdict)

# Step 4: correct only if the audit recommends it.
report = framework.audit(models["LR"], X_source, y_source, X_target, y_target,
                         metric="balanced_accuracy")
if report.recommendation == "correction_required":
    corrected_model, correction = framework.correct(
        models["LR"], X_source, y_source,
        X_target_labeled, y_target_labeled, n_samples=75)
    # Step 5: check the corrected model on target samples it has not seen.
    validation = framework.validate(corrected_model, X_target_holdout, y_target_holdout)

framework.summary()
framework.generate_report("ancestryaudit_report.json")
```

Pass the **full** source cohort to `audit()` and `audit_many()`: they split it
75/25 internally. Passing only a training subset makes the library split it
again.

---

## What the recommendation means

| `recommendation` | Meaning | What to do |
| --- | --- | --- |
| `no_action` | No significant gap above `threshold_pp` (default 2 points) | Check `power_analysis()`: a non-significant result is not proof of fairness |
| `target_more_accurate` | Significant gap **in favor of** the target group | No correction; explain it with `diagnose()` |
| `check_composition` | Significant gap under raw accuracy, but class proportions differ | Rerun with `metric="balanced_accuracy"` |
| `correction_required` | Significant gap against the target group, not explained by class proportions | Check subtypes with `diagnose()`, then consider `correct()` |

`audit_many()` returns one of `no_significant_gap`, `target_less_accurate`,
`target_more_accurate` or `inconsistent_direction`, based on the models that
remain significant after Holm correction.

`correct()` issues a warning if it is called without an audit, or if the last
audit did not recommend correction: fine-tuning with target samples can raise
accuracy even when there is no bias, and that change should not be reported as
a corrected bias.

---

## API reference

### `AncestryAuditFramework(random_state=42, n_permutations=1000, threshold_pp=2.0, threshold_p=0.05)`

| Method | Description | Returns |
| --- | --- | --- |
| `power_analysis(n_source, n_target, expected_gap_pp=3.0)` | Power to detect a gap of the given size, and the sample sizes needed for 80% power | `dict` |
| `diagnose(y_source, y_target, subgroup_source=None, subgroup_target=None)` | Chi-square tests for class proportions and subtype mix | `CompositionReport` |
| `audit(model, X_source, y_source, X_target, y_target, metric="accuracy")` | Permutation test for one model's accuracy gap | `AuditReport` |
| `audit_many(models, X_source, y_source, X_target, y_target, metric="balanced_accuracy")` | Permutation test for several models with Holm correction | `MultiAuditReport` |
| `filter_stratification_noise(X, gene_list, gene_biotype=None)` | Remove olfactory-receptor genes, pseudogenes and uncharacterized loci | `(X_filtered, kept_genes, filter_log)` |
| `correct(model, X_source, y_source, X_target_labeled, y_target_labeled, n_samples=75)` | Fine-tuning with labeled target samples | `(corrected_model, CorrectionReport)` |
| `validate(corrected_model, X_holdout, y_holdout)` | Gap and accuracy before and after correction on held-out data | `ValidationReport` |
| `summary()`, `generate_report(save_path)` | Printed summary and JSON report of all steps run | `str`, `dict` |

Module-level helpers: `check_composition()`, `run_per_algorithm_audit()`,
`holm_bonferroni()`.

### `AuditReport`

| Field | Description |
| --- | --- |
| `gap_pp` | Source minus target score, in percentage points (positive = target worse) |
| `p_value` | Two-sided permutation p-value |
| `cohen_d` | Effect size (accuracy metric only; `None` for balanced accuracy) |
| `null_ci` | 2.5 and 97.5 percentiles of the permutation null (not a confidence interval on the gap) |
| `source_accuracy`, `target_accuracy` | Score on held-out source data and on the target data |
| `n_source`, `n_target` | Sample counts |
| `recommendation` | See the table above |
| `metric` | `"accuracy"` or `"balanced_accuracy"` |
| `class_composition_p` | Chi-square p-value for class proportions |
| `notes` | Plain-language advice |

### `CompositionReport`

`class_props_source`, `class_props_target`, `class_p`, `class_differs`,
`subgroup_labels`, `subgroup_props_source`, `subgroup_props_target`,
`subgroup_p`, `subgroup_differs`, `advice`; full counts in `details`.

### `CorrectionReport`

`delta_pp` (accuracy change on a fixed target holdout), `p_value` (McNemar),
`n_used`, `refit_robustness` (5 refits), `all_positive`,
`direction_confirmed`, `baseline_accuracy`, `corrected_accuracy_holdout`
(use this one for comparisons), `corrected_accuracy`, `per_class_holdout`.

### `ValidationReport`

`pre_gap`, `post_gap`, `correction_magnitude`, `improvement_pp`,
`pre_accuracy_target`, `post_accuracy_target`.

---

## Gene filter

`filter_stratification_noise()` removes three gene categories that mostly
reflect population history rather than cancer biology: olfactory-receptor
genes (`OR*`), pseudogenes (by Ensembl/GENCODE biotype when `gene_biotype` is
given, otherwise by name pattern, with a warning), and uncharacterized
clone-based loci (names containing `.`).

---

## Validation on TCGA data

AncestryAudit was validated on gene-level copy-number data from The Cancer
Genome Atlas. Six classifiers were trained on self-reported White patients to
distinguish liver cancer (LIHC) from stomach cancer (STAD) and tested on
self-reported Asian patients. The full notebook is in
[`validation/`](validation/) and on
[Kaggle](https://www.kaggle.com/code/danae345/project-final).

| Analysis | White / Asian | Significant after Holm (raw / balanced) |
| --- | --- | --- |
| Liver vs stomach | 451 / 242 | 0/6 / 0/6 |
| Negative control: thyroid vs bladder | 636 / 89 | 0/6 / 0/6 |
| Negative control: colon vs lung | 594 / 17 | 0/6 / 0/6 |
| Esophagus vs stomach | 373 / 125 | 0/6 / 2/6, models more accurate for Asian patients |
| Squamous esophagus vs stomach | 307 / 124 | 2/6 / 1/6 |

- **Sensitivity.** At these sample sizes the test detects gaps of about
  11 percentage points with 80% power; at zero true gap it flagged 3.0% of
  simulated datasets (nominal 5%).
- **Subtype composition.** 97% of Asian versus 36% of White esophageal tumors
  were squamous (chi-square = 39.7, p < 0.001). This explained why models
  looked more accurate for Asian patients; balanced accuracy alone did not
  remove the effect. Among squamous tumors only, raw-accuracy gaps of up to
  20 percentage points shrank or disappeared under balanced accuracy.
- **Correction.** Fine-tuning with half of the Asian patients raised accuracy
  on the other half by 2.70 percentage points (95% CI 0.69 to 4.41, raw
  accuracy, 10 seeds). Unsupervised CORAL alignment lowered accuracy for all
  six models.

---

## Limitations

- Groups are defined by self-reported race, a social category, not by genetic
  ancestry. Report gaps as differences between self-reported groups.
- The library checks known causes of a gap (chance, class proportions,
  subtypes). It does not detect batch or sequencing-center effects.
- It trusts the labels it is given: a binary classifier always returns one of
  its two classes, so data from a different task will not be flagged.
- Small minority cohorts limit what can be detected; always report the result
  of `power_analysis()` next to a non-significant audit.
- Validated so far on one data type (copy-number variation) from one source
  (TCGA).

---

## Citation

```
Yergaliyeva, D. (2026). Auditing and improving cancer genomic models for Asian
patients with AncestryAudit. https://github.com/DanYerga/ancestryaudit
```

The validation used data generated by the TCGA Research Network:
https://www.cancer.gov/tcga

---

## License

MIT License. Copyright (c) 2026 Dana Yergaliyeva. See [LICENSE](LICENSE).
