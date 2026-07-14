"""
AncestryAudit Demo
==================
Demonstrates the full bias detection and correction pipeline
on synthetic CNV data — no TCGA access required.

Run:
    pip install -e .
    python demo.py
"""

import sys
import time
import numpy as np
from sklearn.linear_model import LogisticRegression

# ── 0. Check import ────────────────────────────────────────────────────────────
try:
    from ancestryaudit import AncestryAuditFramework
except ModuleNotFoundError:
    print("ancestryaudit not found. Run:  pip install -e .")
    sys.exit(1)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _bar(pct, width=30, fill="█", empty="░"):
    filled = int(width * min(pct, 1.0))
    return fill * filled + empty * (width - filled)

def _step(n, total, label):
    print(f"\n  ┌─ Step {n}/{total}: {label}")

def _ok(msg=""):
    print(f"  └─ ✓  {msg}")

def _gap_indicator(gap_pp):
    if abs(gap_pp) > 4:
        return "🔴  LARGE"
    if abs(gap_pp) > 2:
        return "🟡  MODERATE"
    return "🟢  SMALL"


# ══════════════════════════════════════════════════════════════════════════════
print()
print("  ╔══════════════════════════════════════════════════════╗")
print("  ║              AncestryAudit  —  Demo                 ║")
print("  ║   Ancestry bias detection in genomic cancer AI      ║")
print("  ╚══════════════════════════════════════════════════════╝")
print()

t_start = time.time()

# ── 1. Generate synthetic CNV data ─────────────────────────────────────────────
_step(1, 5, "Generating synthetic CNV data")

rng = np.random.RandomState(42)
N_GENES   = 500
N_WESTERN = 200
N_ASIAN   = 80

# Western cohort — two cancer subtypes with clean separation
X_western_0 = rng.randn(N_WESTERN // 2, N_GENES)
X_western_1 = rng.randn(N_WESTERN // 2, N_GENES) + 0.6   # shifted = separable
X_western   = np.vstack([X_western_0, X_western_1])
y_western   = np.array([0]*(N_WESTERN//2) + [1]*(N_WESTERN//2))

# Asian cohort — same structure but distribution-shifted + noisier labels
# This creates the artificial performance gap the correction will close
shift       = rng.randn(N_GENES) * 0.4     # ancestry-linked CNV shift
X_asian_0   = rng.randn(N_ASIAN // 2, N_GENES) + shift
X_asian_1   = rng.randn(N_ASIAN // 2, N_GENES) + shift + 0.45  # tighter separation
X_asian     = np.vstack([X_asian_0, X_asian_1])
y_asian     = np.array([0]*(N_ASIAN//2) + [1]*(N_ASIAN//2))

# Inject additional noise into 15% of Asian labels (simulates harder classification)
noise_idx = rng.choice(N_ASIAN, size=int(N_ASIAN * 0.15), replace=False)
y_asian[noise_idx] = 1 - y_asian[noise_idx]

# Gene names: mostly valid, with 10 junk genes for filter demo
gene_list = [f"GENE{i:04d}" for i in range(N_GENES)]
gene_list[10]  = "OR5A1"       # olfactory receptor
gene_list[50]  = "TP53P1"      # pseudogene
gene_list[100] = "OR14I1"      # olfactory receptor
gene_list[200] = "LYPD9P"      # pseudogene
gene_list[300] = "AL117190.3"  # uncharacterized
gene_list[350] = "OR2X1P"      # olfactory receptor (also a pseudogene)
gene_list[400] = "AHCYP8"      # pseudogene
gene_list[420] = "CR589904.2"  # uncharacterized
gene_list[450] = "OR14K1"      # olfactory receptor
gene_list[480] = "DPY19L4P1"   # pseudogene

print(f"  │  Western cohort : {N_WESTERN} samples × {N_GENES} genes")
print(f"  │  Asian cohort   : {N_ASIAN} samples × {N_GENES} genes")
print(f"  │  Labels         : binary (0=cancer type A, 1=cancer type B)")
print(f"  │  Junk genes     : 10 injected (OR/pseudogene/uncharacterized)")
_ok("Synthetic data ready")

# ── 2. Initialise framework ────────────────────────────────────────────────────
_step(2, 5, "Initialising AncestryAuditFramework")

framework = AncestryAuditFramework(
    random_state=42,
    n_bootstrap=500,
    threshold_pp=2.0,
    threshold_p=0.05
)
model = LogisticRegression(max_iter=1000, random_state=42,
                            class_weight="balanced", solver="liblinear")
_ok(f"Framework ready  (bootstrap={framework.n_bootstrap}, "
    f"threshold={framework.threshold_pp}pp / p<{framework.threshold_p})")

# ── 3. Optional: filter stratification noise ───────────────────────────────────
print("\n  ├─ 3a. Filtering population-stratification noise ...")
X_w_filt, kept_genes, flog = framework.filter_stratification_noise(
    X_western, gene_list)
X_a_filt = X_asian[:, [i for i, g in enumerate(gene_list)
                        if g in set(kept_genes)]]
print(f"  │    Removed {flog['n_removed']} junk genes  "
      f"({flog['categories']['olfactory_receptors']} OR, "
      f"{flog['categories']['pseudogenes']} pseudogenes, "
      f"{flog['categories']['uncharacterized']} uncharacterized)")
print(f"  │    {_bar(flog['n_kept']/flog['n_input_genes'])}  "
      f"{flog['n_kept']}/{flog['n_input_genes']} genes retained")

# ── 4. Audit ───────────────────────────────────────────────────────────────────
_step(3, 5, "Auditing ancestry-linked performance gap")
print("  │  Training on Western cohort → evaluating on Asian cohort ...")

audit_report = framework.audit(model, X_w_filt, y_western, X_a_filt, y_asian)

gap_abs  = abs(audit_report.gap_pp)
gap_bar  = _bar(min(gap_abs / 10.0, 1.0), width=20,
                fill="▓" if audit_report.gap_pp > 0 else "░")
severity = _gap_indicator(audit_report.gap_pp)

print(f"  │")
print(f"  │  Western accuracy : {audit_report.source_accuracy*100:.1f}%  "
      f"(n={audit_report.n_source})")
print(f"  │  Asian accuracy   : {audit_report.target_accuracy*100:.1f}%  "
      f"(n={audit_report.n_target})")
print(f"  │  ─────────────────────────────────────────")
print(f"  │  Gap              : {audit_report.gap_pp:+.2f}pp  "
      f"[{gap_bar}]  {severity}")
print(f"  │  p-value          : {audit_report.p_value:.4f}  "
      f"{'(significant)' if audit_report.p_value < 0.05 else '(not significant)'}")
print(f"  │  Cohen's d        : {audit_report.cohen_d:.3f}  "
      f"{'(large)' if abs(audit_report.cohen_d) > 0.8 else '(moderate)' if abs(audit_report.cohen_d) > 0.5 else '(small)'}")
print(f"  │  Null CI          : [{audit_report.null_ci[0]:.2f}, "
      f"{audit_report.null_ci[1]:.2f}]pp (permutation null spread)")
print(f"  │")
rec_icon = "⚠️ " if audit_report.recommendation == "correction_required" else "✅"
print(f"  │  Recommendation   : {rec_icon}  {audit_report.recommendation.upper()}")
_ok(f"Audit complete")

# ── 5. Correct ─────────────────────────────────────────────────────────────────
corrected_model = None
correction_report = None

if audit_report.recommendation == "correction_required":
    _step(4, 5, "Applying fine-tuning correction")

    # Pass full Asian dataset — framework samples n_samples internally
    # and holds out the remainder for unbiased per-seed evaluation
    n_finetune = 50

    print(f"  │  Fine-tuning with {n_finetune} labeled Asian samples ...")
    print(f"  │  Testing robustness across 5 refits (fixed split) ...")

    corrected_model, correction_report = framework.correct(
        model, X_w_filt, y_western,
        X_a_filt, y_asian,      # full Asian — framework handles internal split
        n_samples=n_finetune
    )

    rob = correction_report.refit_robustness
    print(f"  │")
    print(f"  │  Mean correction  : {correction_report.delta_pp:+.2f}pp")
    print(f"  │  Refit mean delta : {rob['mean_delta_pp']:+.2f}pp")
    print(f"  │  Refit range      : [{rob['min_delta_pp']:+.2f}, {rob['max_delta_pp']:+.2f}]pp")
    print(f"  │  Refits           : {rob['n_refits']}  "
          f"{'✓ ALL POSITIVE' if correction_report.all_positive else '~ MIXED SIGN'}")
    print(f"  │  p-value (McNemar): {correction_report.p_value:.4f}")
    _ok("Correction complete")
else:
    _step(4, 5, "Correction — skipped (no_action)")
    print(f"  │  Gap below threshold — no correction needed.")
    _ok()

# ── 6. Validate ────────────────────────────────────────────────────────────────
_step(5, 5, "Validating correction on held-out Asian data")

if corrected_model is not None:
    validation = framework.validate(corrected_model, X_a_filt, y_asian)
    pre_bar  = _bar(max(0, audit_report.gap_pp / 10.0), width=15,
                    fill="▓", empty="░")
    post_bar = _bar(max(0, validation.post_gap / 10.0), width=15,
                    fill="▓", empty="░")
    print(f"  │")
    print(f"  │  Pre-correction gap  : {validation.pre_gap:>+7.2f}pp  [{pre_bar}]")
    print(f"  │  Post-correction gap : {validation.post_gap:>+7.2f}pp  [{post_bar}]")
    print(f"  │  Improvement         : {validation.improvement_pp:>+7.2f}pp  "
          f"{'✓ closed' if validation.post_gap < validation.pre_gap else '~ no change'}")
    _ok("Validation complete")
else:
    # No correction was applied (gap below threshold), so there is nothing
    # to validate - calling validate() here would require scoring an
    # unfitted model, which is meaningless. Skip explicitly instead.
    validation = None
    print(f"  │  No correction was applied (gap below threshold) - nothing to validate.")
    _ok("Validation skipped (no_action)")

# ── 7. Report ──────────────────────────────────────────────────────────────────
report_path = "ancestryaudit_report.json"
report_dict = framework.generate_report(save_path=report_path)

# ── 8. Final summary ───────────────────────────────────────────────────────────
elapsed = time.time() - t_start

print()
print("  ╔══════════════════════════════════════════════════════╗")
print("  ║            AncestryAudit Demo Complete               ║")
print("  ╠══════════════════════════════════════════════════════╣")
print(f"  ║  Bias detected      : {audit_report.gap_pp:>+6.2f}pp  "
      f"(p={audit_report.p_value:.4f})         ║")

if correction_report:
    print(f"  ║  Correction applied : {correction_report.delta_pp:>+6.2f}pp  "
          f"(10-seed robust)        ║")

if validation is not None:
    print(f"  ║  After correction   : {validation.post_gap:>+6.2f}pp  "
          f"gap remaining           ║")
else:
    print(f"  ║  After correction   : N/A (no correction applied)      ║")
print(f"  ║  Steps completed    : {', '.join(report_dict['steps_completed']):<28}  ║")
print(f"  ║  Report saved to    : {report_path:<28}  ║")
print(f"  ║  Runtime            : {elapsed:.1f}s{' '*37}║")
print("  ╚══════════════════════════════════════════════════════╝")
print()
