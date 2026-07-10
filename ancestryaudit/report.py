"""
report.py — Full JSON report generation for all pipeline steps.
"""
import json
import datetime
from typing import Optional, Dict, Any


def generate_full_report(
    audit_report=None,
    correction_report=None,
    validation_report=None,
    filter_log=None,
    save_path: str = "ancestryaudit_report.json"
) -> Dict[str, Any]:
    """
    Assemble a full JSON report from completed pipeline steps.

    Parameters
    ----------
    audit_report : AuditReport or None
    correction_report : CorrectionReport or None
    validation_report : ValidationReport or None
    filter_log : dict or None
    save_path : str, output file path

    Returns
    -------
    dict : complete structured report
    """
    report: Dict[str, Any] = {
        "ancestryaudit_version": "0.1.0",
        "generated_at": datetime.datetime.now().isoformat(),
        "steps_completed": [],
    }

    if filter_log is not None:
        report["filter"] = _safe_dict(filter_log)
        report["steps_completed"].append("filter")

    if audit_report is not None:
        report["audit"] = {
            "gap_pp":           round(float(audit_report.gap_pp), 4),
            "p_value":          round(float(audit_report.p_value), 6),
            "cohen_d":          round(float(audit_report.cohen_d), 4),
            "ci_95_lower":      round(float(audit_report.ci_95[0]), 4),
            "ci_95_upper":      round(float(audit_report.ci_95[1]), 4),
            "source_accuracy":  round(float(audit_report.source_accuracy), 4),
            "target_accuracy":  round(float(audit_report.target_accuracy), 4),
            "n_source":         int(audit_report.n_source),
            "n_target":         int(audit_report.n_target),
            "recommendation":   str(audit_report.recommendation),
        }
        report["steps_completed"].append("audit")

    if correction_report is not None:
        report["correction"] = {
            "delta_pp":          round(float(correction_report.delta_pp), 4),
            "p_value":           round(float(correction_report.p_value), 6),
            "n_used":            int(correction_report.n_used),
            "baseline_accuracy": round(float(correction_report.baseline_accuracy), 4),
            "corrected_accuracy":round(float(correction_report.corrected_accuracy), 4),
            "all_positive":      bool(correction_report.all_positive),
            "seed_robustness":   _safe_dict(correction_report.seed_robustness),
        }
        report["steps_completed"].append("correction")

    if validation_report is not None:
        report["validation"] = {
            "pre_gap":              round(float(validation_report.pre_gap), 4),
            "post_gap":             round(float(validation_report.post_gap), 4),
            "correction_magnitude": round(float(validation_report.correction_magnitude), 4),
            "improvement_pp":       round(float(validation_report.improvement_pp), 4),
            "pre_accuracy_target":  round(float(validation_report.pre_accuracy_target), 4),
            "post_accuracy_target": round(float(validation_report.post_accuracy_target), 4),
        }
        report["steps_completed"].append("validation")

    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def _safe_dict(d):
    """Recursively convert dict values to JSON-safe types."""
    if isinstance(d, dict):
        return {k: _safe_dict(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_safe_dict(v) for v in d]
    if hasattr(d, "item"):   # numpy scalar
        return d.item()
    return d
