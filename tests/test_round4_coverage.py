"""
test_round4_coverage.py
========================
Regression tests for previously-untested surfaces: validate(),
gene_biotype-based filtering, and metric="balanced_accuracy" in audit().

These existed as code before this test file but had zero test coverage,
flagged in a round-4 audit. Each test targets the specific bug that audit
found and was subsequently fixed, so a regression would be caught here.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.linear_model import LogisticRegression
from ancestryaudit import AncestryAuditFramework
from ancestryaudit.filter import filter_noise, _is_pseudogene, _is_uncharacterized


def test_validate_uses_same_holdout_for_pre_and_post():
    """
    Regression test for the validate() sample-mismatch bug: pre_gap and
    post_gap must be computed on the SAME X_holdout/y_holdout, not a
    mismatched pair (original bug: pre_gap used the audit()-time full
    target set, post_gap used a different user-supplied holdout).
    """
    rng = np.random.RandomState(3)
    X_s = rng.randn(200, 15); y_s = (X_s[:, 0] + X_s[:, 1] > 0).astype(int)
    X_t = rng.randn(100, 15); y_t = (X_t[:, 0] + X_t[:, 1] > 0).astype(int)

    fw = AncestryAuditFramework()
    fw.audit(LogisticRegression(max_iter=500), X_s, y_s, X_t, y_t)
    corrected_model, _ = fw.correct(
        LogisticRegression(max_iter=500), X_s, y_s, X_t, y_t, n_samples=30)

    X_holdout, y_holdout = X_t[:20], y_t[:20]
    val = fw.validate(corrected_model, X_holdout, y_holdout)

    # correction_magnitude and improvement_pp must be self-consistent
    # (this identity broke under the old mismatched-sample-set bug)
    assert abs(val.correction_magnitude - val.improvement_pp) < 1e-9, (
        "correction_magnitude and improvement_pp diverged - "
        "pre_gap/post_gap likely computed on different sample sets again."
    )
    print("PASS: validate() pre/post use consistent holdout.")


def test_gene_biotype_overrides_name_heuristic():
    """
    Regression test: a gene_biotype annotation confirming a real biotype
    (e.g. protein_coding) must override BOTH the pseudogene name-pattern
    heuristic AND the uncharacterized-loci dot-heuristic. Covers the two
    real bugs found in rounds 2-4: exact-biotype-string mismatch, and
    biotype not overriding the "." dot-check for versioned Ensembl IDs.
    """
    genes = ["TP53", "NUP98", "ENSG00000141510.16", "TP53P1"]
    biotype = {
        "TP53": "protein_coding",
        "NUP98": "protein_coding",
        "ENSG00000141510.16": "protein_coding",
        "TP53P1": "processed_pseudogene",
    }
    X = np.random.randn(5, len(genes))
    _, kept, flog = filter_noise(X, genes, gene_biotype=biotype)

    assert set(kept) == {"TP53", "NUP98", "ENSG00000141510.16"}, (
        f"Expected 3 real genes kept, got {kept}"
    )
    assert flog["categories"]["pseudogenes"] == 1
    assert flog["categories"]["uncharacterized"] == 0
    print("PASS: gene_biotype correctly overrides both heuristics.")


def test_audit_balanced_accuracy_reachable_and_correct():
    """
    Regression test: metric="balanced_accuracy" must be reachable through
    the public AncestryAuditFramework.audit() API (round-4 bug: the
    balanced_accuracy path existed in compute_audit() but audit() had no
    way to request it), and cohen_d must be None on that path.
    """
    rng = np.random.RandomState(5)
    X_s = rng.randn(200, 15); y_s = (X_s[:, 0] + X_s[:, 1] > 0).astype(int)
    X_t = rng.randn(100, 15); y_t = (X_t[:, 0] + X_t[:, 1] > 0).astype(int)

    fw = AncestryAuditFramework()
    report = fw.audit(LogisticRegression(max_iter=500), X_s, y_s, X_t, y_t,
                       metric="balanced_accuracy")

    assert report.metric == "balanced_accuracy"
    assert report.cohen_d is None
    assert isinstance(report.gap_pp, float)

    report_dict = fw.generate_report("/tmp/_test_round4_report.json")
    assert report_dict["audit"]["cohen_d"] is None, (
        "cohen_d should serialize as null, not crash on float(None)"
    )
    print("PASS: balanced_accuracy reachable via audit(), cohen_d handled.")


def test_uncharacterized_loci_detected_by_name_not_biotype():
    """
    Regression test for a self-audit finding: an earlier fix let biotype
    override the uncharacterized-loci check, trusting biotype=="TEC" as
    the sole signal. That wrongly KEPT real clone-based loci (e.g. this
    module's own docstring example, "AL117190.3") whenever their biotype
    was anything else (lncRNA, protein_coding, etc.) - common in real
    GENCODE data. Detection must be name-pattern based, with only a
    versioned-Ensembl-ID exception, regardless of biotype.
    """
    genes = ["AL117190.3", "RP11-34P13.7", "CR589904.2",
             "TP53", "ENSG00000141510.16"]
    biotype = {
        "AL117190.3": "lncRNA",
        "RP11-34P13.7": "processed_transcript",
        "CR589904.2": "antisense",
        "TP53": "protein_coding",
        "ENSG00000141510.16": "protein_coding",
    }
    X = np.random.randn(5, len(genes))
    _, kept, flog = filter_noise(X, genes, gene_biotype=biotype)

    assert set(kept) == {"TP53", "ENSG00000141510.16"}, (
        f"Expected only TP53 and the versioned Ensembl ID kept, got {kept}"
    )
    assert flog["categories"]["uncharacterized"] == 3
    print("PASS: uncharacterized loci detected by name pattern regardless of biotype.")


if __name__ == "__main__":
    test_validate_uses_same_holdout_for_pre_and_post()
    test_gene_biotype_overrides_name_heuristic()
    test_audit_balanced_accuracy_reachable_and_correct()
    test_uncharacterized_loci_detected_by_name_not_biotype()
    print("\nAll round-4 coverage tests passed.")
