"""
diagnose.py — Check whether two cohorts differ in composition.

An accuracy gap between two populations does not by itself mean that the
model is biased. The gap can also come from the make-up of the groups:

* Class proportions. Raw accuracy averages over classes, so if a model
  separates one class more easily than the other, the group with more of
  the easier class scores higher even when the model treats every sample
  the same way. Balanced accuracy removes this effect.
* Subtypes within a class. Balanced accuracy does NOT remove differences
  in subtype mix within a class (for example squamous versus
  adenocarcinoma esophageal tumors). These have to be checked directly.

check_composition() compares both with chi-square tests and returns plain
advice on what to do before any correction.
"""
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy import stats


def _counts(values: Sequence, categories: Sequence) -> np.ndarray:
    values = list(values)
    return np.array([sum(1 for v in values if v == c) for c in categories],
                    dtype=int)


def _clean_labels(values: Optional[Sequence]):
    """Drop missing subgroup labels (None, NaN, empty string)."""
    if values is None:
        return None
    cleaned = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        cleaned.append(v)
    return cleaned


def _chi2(table: np.ndarray):
    """Chi-square test on a 2 x k table; drops empty columns."""
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2 or (table.sum(axis=1) == 0).any():
        return None, None
    chi2, p, _, _ = stats.chi2_contingency(table)
    return float(chi2), float(p)


def check_composition(
    y_source,
    y_target,
    subgroup_source: Optional[Sequence] = None,
    subgroup_target: Optional[Sequence] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Test whether two cohorts differ in class proportions and, optionally,
    in subtype mix.

    Parameters
    ----------
    y_source, y_target : array-like
        Class labels of the source and target cohorts.
    subgroup_source, subgroup_target : sequence, optional
        Subtype label for each sample in a class of interest (for example
        "squamous" / "adenocarcinoma" for esophageal tumors). Missing values
        (None, NaN, "") are dropped. Both must be given to run the check.
    alpha : float, default 0.05
        Significance level for both chi-square tests.

    Returns
    -------
    dict with keys:
        class_labels, class_counts_source, class_counts_target,
        class_props_source, class_props_target,
        class_chi2, class_p, class_differs,
        subgroup_labels, subgroup_counts_source, subgroup_counts_target,
        subgroup_props_source, subgroup_props_target,
        subgroup_chi2, subgroup_p, subgroup_differs,
        advice : list of str
    """
    y_source = np.asarray(y_source)
    y_target = np.asarray(y_target)
    labels = sorted(set(y_source.tolist()) | set(y_target.tolist()))

    cs = _counts(y_source.tolist(), labels)
    ct = _counts(y_target.tolist(), labels)
    class_chi2, class_p = _chi2(np.vstack([cs, ct]))
    class_differs = bool(class_p is not None and class_p < alpha)

    result: Dict[str, Any] = {
        "class_labels": [l.item() if hasattr(l, "item") else l for l in labels],
        "class_counts_source": cs.tolist(),
        "class_counts_target": ct.tolist(),
        "class_props_source": (cs / cs.sum()).round(4).tolist(),
        "class_props_target": (ct / ct.sum()).round(4).tolist(),
        "class_chi2": class_chi2,
        "class_p": class_p,
        "class_differs": class_differs,
        "subgroup_labels": None,
        "subgroup_counts_source": None,
        "subgroup_counts_target": None,
        "subgroup_props_source": None,
        "subgroup_props_target": None,
        "subgroup_chi2": None,
        "subgroup_p": None,
        "subgroup_differs": None,
    }

    advice = []
    if class_differs:
        advice.append(
            "Class proportions differ between cohorts "
            f"(chi2={class_chi2:.2f}, p={class_p:.2g}). Raw accuracy is "
            "affected by class mix: audit with metric='balanced_accuracy'.")

    sub_s = _clean_labels(subgroup_source)
    sub_t = _clean_labels(subgroup_target)
    if sub_s is not None and sub_t is not None and sub_s and sub_t:
        sub_labels = sorted(set(sub_s) | set(sub_t), key=str)
        ss = _counts(sub_s, sub_labels)
        st = _counts(sub_t, sub_labels)
        sub_chi2, sub_p = _chi2(np.vstack([ss, st]))
        sub_differs = bool(sub_p is not None and sub_p < alpha)
        result.update({
            "subgroup_labels": sub_labels,
            "subgroup_counts_source": ss.tolist(),
            "subgroup_counts_target": st.tolist(),
            "subgroup_props_source": (ss / ss.sum()).round(4).tolist(),
            "subgroup_props_target": (st / st.sum()).round(4).tolist(),
            "subgroup_chi2": sub_chi2,
            "subgroup_p": sub_p,
            "subgroup_differs": sub_differs,
        })
        if sub_differs:
            advice.append(
                "Subtype mix differs between cohorts "
                f"(chi2={sub_chi2:.2f}, p={sub_p:.2g}). Balanced accuracy "
                "does not remove this: repeat the audit within a single "
                "subtype before attributing a gap to ancestry.")
        if min(ss.sum(), st.sum()) < 20:
            advice.append(
                "Fewer than 20 samples with a known subtype in one cohort: "
                "the subtype test has low power.")

    if not advice:
        advice.append(
            "No composition difference detected. A gap, if significant, is "
            "not explained by class proportions"
            + (" or subtype mix." if result["subgroup_p"] is not None
               else "; subtype mix was not checked."))

    result["advice"] = advice
    return result
