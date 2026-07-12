"""
filter.py — Remove ancestry-linked CNV regions unrelated to cancer biology.

Filters:
  - Olfactory receptor genes (OR*)      — known population-stratification markers
  - Pseudogenes (*P, *P1, *P2, …)       — non-functional, ancestry-variant
  - Uncharacterized clone-based loci    — names containing '.' (e.g. AL117190.3)

Reference: Kaufman et al. (2012) on feature-space snooping disclosure.
"""
import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union


def filter_noise(
    X: Union[np.ndarray, "pd.DataFrame"],
    gene_list: List[str]
) -> Tuple[Union[np.ndarray, "pd.DataFrame"], List[str], Dict]:
    """
    Remove ancestry-linked CNV regions unrelated to cancer biology.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame, shape (n_samples, n_genes)
        CNV feature matrix. Columns must correspond to gene_list.
    gene_list : list of str
        Gene names for each column of X.

    Returns
    -------
    X_filtered : same type as X, with excluded columns removed
    kept_genes : list of str
    filter_log : dict with removal statistics and rationale
    """
    gene_list = list(gene_list)
    if len(gene_list) != _n_cols(X):
        raise ValueError(
            f"gene_list length ({len(gene_list)}) must match "
            f"number of columns in X ({_n_cols(X)})."
        )

    keep_mask = np.array([not _is_junk(g) for g in gene_list])

    removed = [g for g, k in zip(gene_list, keep_mask) if not k]
    kept    = [g for g, k in zip(gene_list, keep_mask) if k]

    or_genes     = [g for g in removed if re.match(r"^OR\d", g)]
    pseudogenes  = [g for g in removed if _is_pseudogene(g)]
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

def _is_junk(name: str) -> bool:
    """Return True if gene should be excluded."""
    if not isinstance(name, str) or not name.strip():
        return True
    if re.match(r"^OR\d", name):          # Olfactory receptors
        return True
    if _is_pseudogene(name):              # Pseudogenes
        return True
    if "." in name:                        # Clone-based uncharacterized loci
        return True
    return False


def _is_pseudogene(name: str) -> bool:
    """Pseudogene: HGNC convention — trailing P optionally followed by digit."""
    return bool(re.search(r"P\d*$", name))


def _n_cols(X) -> int:
    if isinstance(X, pd.DataFrame):
        return X.shape[1]
    return np.asarray(X).shape[1]
