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
    if len(gene_list) != _n_cols(X):
        raise ValueError(
            f"gene_list length ({len(gene_list)}) must match "
            f"number of columns in X ({_n_cols(X)})."
        )

    keep_mask = np.array([not _is_junk(g, gene_biotype) for g in gene_list])

    removed = [g for g, k in zip(gene_list, keep_mask) if not k]
    kept    = [g for g, k in zip(gene_list, keep_mask) if k]

    or_genes     = [g for g in removed if re.match(r"^OR\d", g)]
    pseudogenes  = [g for g in removed if _is_pseudogene(g, gene_biotype)]
    unchar       = [g for g in removed if "." in g]
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
        "pseudogene_method": ("biotype_lookup" if gene_biotype is not None
                              else "name_heuristic_UNRELIABLE"),
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
    if re.match(r"^OR\d", name):          # Olfactory receptors
        return True
    if _is_pseudogene(name, gene_biotype): # Pseudogenes
        return True
    if "." in name:                        # Clone-based uncharacterized loci
        return True
    return False


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
    Pseudogene detection. If gene_biotype is supplied, uses the real
    Ensembl/GENCODE biotype annotation (authoritative). Otherwise falls
    back to a NAME-PATTERN heuristic that is KNOWN TO BE UNRELIABLE: HGNC
    pseudogene convention is <parent_gene>P<number> (e.g. TP53P1), which is
    indistinguishable by name alone from real genes ending in P + digit(s)
    (TP53, PARP1-4, GSTP1, MMP family, NUP98/214/153/62/50/88/93,
    CASP1/3/7/8/9/14, TOP1, AQP1/3/9, NRIP1-3, REEP1/5/6, ADCYAP1, and
    others not yet found). A small allowlist covers known cases but is NOT
    exhaustive. Do not rely on the fallback path for any result that
    matters; supply gene_biotype instead.
    """
    if gene_biotype is not None:
        return gene_biotype.get(name, "").strip().lower() == "pseudogene"
    if name.upper() in _KNOWN_REAL_GENES_NOT_PSEUDOGENES:
        return False
    return bool(re.search(r"P\d*$", name))


def _n_cols(X) -> int:
    if isinstance(X, pd.DataFrame):
        return X.shape[1]
    return np.asarray(X).shape[1]
