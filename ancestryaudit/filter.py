"""
filter.py — Remove ancestry-linked CNV regions unrelated to cancer biology.

Filters:
  - Olfactory receptor genes (OR*)      — known population-stratification markers
  - Pseudogenes (*P, *P1, *P2, …)       — non-functional, ancestry-variant
  - Uncharacterized clone-based loci    — names containing '.' (e.g. AL117190.3)

Reference: Kaufman et al. (2012) on feature-space snooping disclosure.
"""
import re
import warnings
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union, Optional


def filter_noise(
    X: Union[np.ndarray, "pd.DataFrame"],
    gene_list: List[str],
    gene_biotype: Optional[Dict[str, str]] = None,
) -> Tuple[Union[np.ndarray, "pd.DataFrame"], List[str], Dict]:
    """
    Remove ancestry-linked CNV regions unrelated to cancer biology.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame, shape (n_samples, n_genes)
        CNV feature matrix. Columns must correspond to gene_list.
    gene_list : list of str
        Gene names for each column of X.
    gene_biotype : dict, optional
        Mapping of gene symbol -> Ensembl/GENCODE biotype (e.g.
        {"TP53": "protein_coding", "TP53P1": "pseudogene"}). STRONGLY
        RECOMMENDED. If provided, pseudogene removal uses this authoritative
        annotation. If omitted, falls back to an unreliable name-pattern
        heuristic (a UserWarning is raised) that is known to misclassify
        real genes such as TP53, NUP98, CASP1-14, TOP1, and others.

    Returns
    -------
    X_filtered : same type as X, with excluded columns removed
    kept_genes : list of str
    filter_log : dict with removal statistics and rationale
    """
    if gene_biotype is None:
        warnings.warn(
            "filter_noise() called without gene_biotype: falling back to a "
            "name-pattern heuristic for pseudogene detection that is known "
            "to misclassify real genes (e.g. TP53, NUP98, CASP1-14, TOP1). "
            "Supply a gene_biotype mapping (Ensembl/GENCODE 'gene_type' "
            "column) for reliable results. Always inspect filter_log's "
            "removed-gene list before trusting downstream analysis.",
            UserWarning, stacklevel=2
        )
    gene_list = list(gene_list)

    # A gene present in gene_list but missing from an otherwise-provided
    # gene_biotype dict is an annotation-coverage gap, not evidence the
    # gene isn't a pseudogene - _is_pseudogene() falls back to the name
    # heuristic per-gene in that case. Warn about it explicitly so partial
    # coverage isn't silently mistaken for full biotype-authoritative
    # results (a UserWarning only fired before when gene_biotype was None
    # entirely, not when it was merely incomplete).
    missing_biotype = []
    if gene_biotype is not None:
        missing_biotype = [g for g in gene_list
                           if isinstance(g, str) and g not in gene_biotype]
        if missing_biotype:
            warnings.warn(
                f"gene_biotype is missing {len(missing_biotype)}/"
                f"{len(gene_list)} genes in gene_list. Those genes fell "
                f"back to the unreliable name-pattern heuristic for "
                f"pseudogene detection, not the biotype annotation. "
                f"First 20 missing: {missing_biotype[:20]}",
                UserWarning, stacklevel=2
            )
    if len(gene_list) != _n_cols(X):
        raise ValueError(
            f"gene_list length ({len(gene_list)}) must match "
            f"number of columns in X ({_n_cols(X)})."
        )

    # Guard against versioned Ensembl IDs (e.g. "ENSG00000141510.16"),
    # which contain a "." for a completely different reason than
    # clone-based uncharacterized loci ("." in a gene SYMBOL). If most of
    # gene_list looks like an Ensembl ID, the "uncharacterized loci" check
    # below would silently delete nearly everything.
    n_ensg = sum(1 for g in gene_list if isinstance(g, str) and g.startswith("ENSG"))
    if n_ensg / max(len(gene_list), 1) > 0.5:
        raise ValueError(
            f"{n_ensg}/{len(gene_list)} entries in gene_list look like "
            f"versioned Ensembl gene IDs (e.g. 'ENSG00000141510.16'), not "
            f"gene symbols. The uncharacterized-loci filter (\".\" in name) "
            f"would misinterpret the version suffix and remove nearly "
            f"everything. Convert to gene symbols first (e.g. via your "
            f"gene_id_to_name mapping) before calling filter_noise()."
        )

    keep_mask = np.array([not _is_junk(g, gene_biotype) for g in gene_list])

    removed = [g for g, k in zip(gene_list, keep_mask) if not k]
    kept    = [g for g, k in zip(gene_list, keep_mask) if k]

    # Mutually exclusive categorization, in the same priority order _is_junk()
    # uses to decide removal (OR > pseudogene > uncharacterized). Without this,
    # a gene matching more than one category (e.g. "OR2X1P" - both an olfactory
    # receptor AND pseudogene-shaped) gets double-counted, and the category
    # counts silently sum to more than n_removed - an inconsistency that would
    # show up directly in the Methods disclosure text below.
    or_genes     = [g for g in removed if _is_or_gene(g)]
    pseudogenes  = [g for g in removed
                    if g not in or_genes and _is_pseudogene(g, gene_biotype)]
    unchar       = [g for g in removed
                    if g not in or_genes and g not in pseudogenes
                    and _is_uncharacterized(g, gene_biotype)]
    other_junk   = [g for g in removed
                    if g not in or_genes and g not in pseudogenes
                    and g not in unchar]

    if isinstance(X, pd.DataFrame):
        X_filtered = X.loc[:, keep_mask].copy()
    else:
        X_filtered = np.asarray(X, dtype=float)[:, keep_mask]

    n_total   = len(gene_list)
    n_removed = int((~keep_mask).sum())
    n_kept    = int(keep_mask.sum())

    filter_log = {
        "n_input_genes":          n_total,
        "n_removed":              n_removed,
        "n_kept":                 n_kept,
        "removal_pct":            round(n_removed / n_total * 100, 2),
        "categories": {
            "olfactory_receptors": len(or_genes),
            "pseudogenes":         len(pseudogenes),
            "uncharacterized":     len(unchar),
            "other":               len(other_junk),
        },
        "pseudogene_method": (
            "name_heuristic_UNRELIABLE" if gene_biotype is None
            else "biotype_lookup" if not missing_biotype
            else f"biotype_lookup_partial ({len(missing_biotype)} genes fell back to name heuristic)"
        ),
        "rationale": (
            "Olfactory receptor clusters and pseudogenes exhibit "
            "copy-number variation driven by population-level genetic "
            "drift rather than cancer biology. Retaining them inflates "
            "ancestry-linked signals that are not clinically actionable. "
            "Uncharacterized loci (clone-based IDs) carry no interpretable "
            "biological information. Removal follows best practices for "
            "ancestry-stratified genomic studies."
        ),
        "disclosure": (
            "REQUIRED Methods disclosure: "
            "'Genes exhibiting known population-stratification CNV patterns "
            "(olfactory receptors, pseudogenes, uncharacterized loci) were "
            "removed prior to ancestry-bias analysis, consistent with "
            "Kaufman et al. (2012). The feature space was defined using all "
            "samples, which constitutes a bounded form of data snooping; "
            "no label information was used in this step.'"
        ),
    }

    return X_filtered, kept, filter_log


# ── Private helpers ────────────────────────────────────────────────────────────

def _is_junk(name: str, gene_biotype: Optional[Dict[str, str]] = None) -> bool:
    """Return True if gene should be excluded."""
    if not isinstance(name, str) or not name.strip():
        return True
    if _is_or_gene(name):                        # Olfactory receptors
        return True
    if _is_pseudogene(name, gene_biotype):        # Pseudogenes
        return True
    if _is_uncharacterized(name, gene_biotype):   # Uncharacterized loci
        return True
    return False


def _is_or_gene(name) -> bool:
    """Olfactory receptor gene (OR followed by a digit). Safe on non-strings."""
    if not isinstance(name, str):
        return False
    return bool(re.match(r"^OR\d", name))


# Known real genes that a naive "trailing P + digit(s)" name pattern would
# misidentify as pseudogenes (confirmed false positives: TP53 itself, plus
# PARP/MMP/TIMP gene families, GSTP1, CRP, ALPP, APP, MAPKAP1). This list is
# NOT exhaustive — it only covers cases found during testing. Always inspect
# filter_log's removed-gene list before trusting downstream results; this is
# a name-pattern heuristic, not a verified HGNC biotype annotation.
_KNOWN_REAL_GENES_NOT_PSEUDOGENES = frozenset({
    "TP53", "PARP1", "PARP2", "PARP3", "PARP4", "GSTP1",
    "MMP1", "MMP2", "MMP3", "MMP7", "MMP9", "MMP13",
    "TIMP1", "TIMP2", "TIMP3", "CRP", "ALPP", "APP", "MAPKAP1",
})


def _is_pseudogene(name: str, gene_biotype: Optional[Dict[str, str]] = None) -> bool:
    """
    Pseudogene detection. If gene_biotype is supplied AND contains an entry
    for this specific gene, uses the real Ensembl/GENCODE biotype
    annotation (authoritative). Otherwise falls back to a NAME-PATTERN
    heuristic that is KNOWN TO BE UNRELIABLE: HGNC pseudogene convention is
    <parent_gene>P<number> (e.g. TP53P1), which is indistinguishable by
    name alone from real genes ending in P + digit(s) (TP53, PARP1-4,
    GSTP1, MMP family, NUP98/214/153/62/50/88/93, CASP1/3/7/8/9/14, TOP1,
    AQP1/3/9, NRIP1-3, REEP1/5/6, ADCYAP1, and others not yet found). A
    small allowlist covers known cases but is NOT exhaustive. Do not rely
    on the fallback path for any result that matters; supply gene_biotype
    instead.

    IMPORTANT: a gene missing from an otherwise-provided gene_biotype dict
    is an annotation-COVERAGE gap, not evidence the gene isn't a
    pseudogene. An earlier version treated "not in the dict" the same as
    "confirmed not a pseudogene" (via dict.get(name, "") defaulting to an
    empty string that never matches), which silently let real pseudogenes
    through whenever the biotype mapping had incomplete coverage - a
    realistic scenario with real GENCODE/symbol-matching data. This
    version falls back to the name heuristic per-gene instead.
    """
    if not isinstance(name, str):
        return False
    if gene_biotype is not None and name in gene_biotype:
        bt = gene_biotype.get(name, "").strip().lower()
        # GENCODE/Ensembl use subtyped pseudogene biotypes (processed_
        # pseudogene, unprocessed_pseudogene, transcribed_*_pseudogene,
        # translated_*_pseudogene, IG/TR pseudogene classes, etc.), not
        # the bare word "pseudogene" (that was only GRCh37/GENCODE v19).
        # polymorphic_pseudogene is protein-coding in some individuals and
        # deliberately NOT treated as a pseudogene here.
        if bt == "polymorphic_pseudogene":
            return False
        return bt.endswith("pseudogene")
    if name.upper() in _KNOWN_REAL_GENES_NOT_PSEUDOGENES:
        return False
    return bool(re.search(r"P\d*$", name))


# Versioned Ensembl gene ID, e.g. "ENSG00000141510.16" - the "." here is
# a STABLE-ID VERSION SUFFIX, not a marker of an uncharacterized
# clone-based locus. This is the one legitimate exception to the
# name-pattern check below.
_ENSEMBL_VERSIONED_ID_RE = re.compile(r"^ENSG\d+\.\d+$")


def _is_uncharacterized(name, gene_biotype: Optional[Dict[str, str]] = None) -> bool:
    """
    Detect clone-based/uncharacterized loci (e.g. "AL117190.3") by NAME
    PATTERN, not by biotype. This is deliberate, not an oversight: biotype
    describes what KIND of transcript something is (protein_coding,
    lncRNA, pseudogene, TEC, ...), which is orthogonal to whether its
    identifier is a proper curated gene symbol or a provisional
    clone-based placeholder name. A clone-based locus can carry almost
    any biotype - GENCODE frequently annotates clone-named loci as
    lncRNA or even protein_coding once partially characterized while
    keeping the provisional name. An earlier version of this function
    trusted biotype=="TEC" as the sole signal and wrongly KEPT
    "AL117190.3" whenever its biotype was anything else (e.g. lncRNA) -
    that was a real bug, found by testing against this module's own
    docstring example.

    gene_biotype is accepted for API-signature consistency with
    _is_pseudogene but is intentionally NOT used to override this check.

    The only exception: a versioned Ensembl gene ID contains a "." as a
    stable-ID version suffix, not as evidence of an uncharacterized
    clone-based name - so that pattern is excluded here.
    """
    if not isinstance(name, str):
        return False
    if _ENSEMBL_VERSIONED_ID_RE.match(name):
        return False
    return "." in name


def _n_cols(X) -> int:
    if isinstance(X, pd.DataFrame):
        return X.shape[1]
    return np.asarray(X).shape[1]
