# Changelog

## 0.4.0 (October 2026)

Diagnose before correcting.

- **New: `diagnose()` / `check_composition()`.** Chi-square tests for class
  proportions and, when subtype labels are given, subtype mix between the two
  cohorts, with plain-language advice.
- **New: `audit_many()`.** Audits several models at once with the Holm
  step-down correction and a direction-aware verdict
  (`no_significant_gap`, `target_less_accurate`, `target_more_accurate`,
  `inconsistent_direction`).
- **Changed: `audit()` recommendation now depends on the direction of the gap.**
  A significant gap in favor of the target group returns
  `target_more_accurate` (previously `correction_required`). A significant gap
  under raw accuracy when class proportions differ returns
  `check_composition`. `AuditReport` gains `class_composition_p` and `notes`.
- **Changed: `correct()` warns** when it is called without an audit, or when
  the last audit did not recommend correction.
- `summary()`, `generate_report()` and the gap plot show the new results.
- `demo.py` adds a diagnosis step and reports non-significant gaps honestly.
- Added the MIT `LICENSE` file, `project_urls` and the validation notebook
  folder. 26 automated tests (12 new).

## 0.3.10 and earlier (July–September 2026)

- Permutation test for the accuracy gap (replacing an invalid bootstrap test).
- `metric="balanced_accuracy"` option in `audit()`.
- `run_per_algorithm_audit()` and `holm_bonferroni()` in `ancestryaudit.audit`.
- Simulation-based `power_analysis()` with calibrated label flips.
- Fine-tuning correction with McNemar test, fixed holdout and per-class report.
- Gene filter for olfactory-receptor genes, pseudogenes and uncharacterized
  loci, with biotype support.
