"""
visualize.py — Publication-quality SVG figures for audit and correction results.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional, List


def plot_gap(audit_report, save_path: Optional[str] = None,
             title: str = "Ancestry Performance Gap") -> "plt.Figure":
    """
    Bar chart: source vs target accuracy with gap annotation.

    Parameters
    ----------
    audit_report : AuditReport
    save_path : str or None; if provided, saves as SVG
    title : str

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    accs   = [audit_report.source_accuracy * 100,
               audit_report.target_accuracy * 100]
    labels = ["Source (Western)", "Target (Asian)"]
    colors = ["#2563eb", "#f97316"]

    ax.bar(labels, accs, color=colors, alpha=0.85,
           edgecolor="black", linewidth=0.5)

    mid_y = (accs[0] + accs[1]) / 2
    ax.annotate(
        f"Gap: {audit_report.gap_pp:+.2f}pp\n"
        f"p={audit_report.p_value:.4f}  d={audit_report.cohen_d:.2f}\n"
        f"Null CI [{audit_report.null_ci[0]:.2f}, {audit_report.null_ci[1]:.2f}]pp (permutation null spread)",
        xy=(0.5, mid_y), xycoords=("axes fraction", "data"),
        ha="center", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="gray", alpha=0.9)
    )

    rec_color = ("#dc2626" if audit_report.recommendation == "correction_required"
                 else "#16a34a")
    ax.text(0.5, 0.97, audit_report.recommendation,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, fontweight="bold", color=rec_color)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, 112)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.show()
    return fig


def plot_correction(pre_accuracy: float, post_accuracy: float,
                    baseline: float,
                    save_path: Optional[str] = None) -> "plt.Figure":
    """
    Before/after correction bar chart.

    Parameters
    ----------
    pre_accuracy : float, target accuracy before correction
    post_accuracy : float, target accuracy after correction
    baseline : float, source accuracy (reference line)
    save_path : str or None
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    categories = ["White-only\n(baseline)", "Fine-tuned\n(corrected)"]
    values     = [pre_accuracy * 100, post_accuracy * 100]
    colors     = ["#dc2626", "#16a34a"]

    ax.bar(categories, values, color=colors, alpha=0.85,
           edgecolor="black", linewidth=0.5, width=0.5)
    ax.axhline(baseline * 100, color="gray", linestyle="--",
               linewidth=1.2, label=f"Source baseline ({baseline*100:.1f}%)")

    delta = (post_accuracy - pre_accuracy) * 100
    ax.text(0.5, (pre_accuracy * 100 + post_accuracy * 100) / 2,
            f"Δ = {delta:+.2f}pp",
            ha="center", fontsize=10, fontweight="bold",
            color="#16a34a" if delta > 0 else "#dc2626")

    ax.set_ylabel("Target (Asian) Accuracy (%)")
    ax.set_title("Fine-tuning Correction: Before vs After")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 112)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.show()
    return fig


def plot_learning_curve(
    n_values: List[int],
    means: List[float],
    sds: List[float],
    baseline: float,
    stabilization_n: Optional[int] = None,
    save_path: Optional[str] = None,
    title: str = "Learning Curve: Sample Size vs Correction Accuracy"
) -> "plt.Figure":
    """
    Learning curve with ±1 SD shaded band.

    Parameters
    ----------
    n_values : list of int, sample sizes tested
    means : list of float, mean accuracy at each n
    sds : list of float, SD at each n
    baseline : float, source-only baseline accuracy
    stabilization_n : int or None, vertical line position
    save_path : str or None
    title : str
    """
    means = np.array(means)
    sds   = np.array(sds)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(n_values, means * 100, marker="o", color="#2563eb",
            linewidth=2, markersize=6, label="Fine-tuned (mean)")
    ax.fill_between(n_values,
                    (means - sds) * 100,
                    (means + sds) * 100,
                    alpha=0.2, color="#2563eb", label="±1 SD")
    ax.axhline(baseline * 100, color="#dc2626", linestyle="--",
               linewidth=1.5, label=f"Baseline ({baseline*100:.2f}%)")

    if stabilization_n is not None:
        ax.axvline(stabilization_n, color="#16a34a", linestyle=":",
                   linewidth=1.5, label=f"Stabilization (n≈{stabilization_n})")

    ax.set_xlabel("Labeled target samples used for fine-tuning")
    ax.set_ylabel("Accuracy on target holdout (%)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.show()
    return fig


def plot_full_pipeline(audit_report, correction_report,
                       validation_report,
                       save_path: Optional[str] = None) -> "plt.Figure":
    """
    Three-panel summary: gap, correction, and validation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("AncestryAudit Pipeline Summary",
                 fontsize=12, fontweight="bold")

    # Panel 1: audit gap
    ax = axes[0]
    bars = ax.bar(["Source", "Target"],
                  [audit_report.source_accuracy * 100,
                   audit_report.target_accuracy * 100],
                  color=["#2563eb", "#f97316"], alpha=0.85,
                  edgecolor="black", linewidth=0.5)
    ax.set_title(f"Audit\nGap={audit_report.gap_pp:+.2f}pp "
                 f"(p={audit_report.p_value:.3f})")
    ax.set_ylabel("Accuracy (%)"); ax.grid(axis="y", alpha=0.3)

    # Panel 2: correction
    ax2 = axes[1]
    ax2.bar(["Pre-correction", "Post-correction"],
            [validation_report.pre_accuracy_target * 100,
             validation_report.post_accuracy_target * 100],
            color=["#dc2626", "#16a34a"], alpha=0.85,
            edgecolor="black", linewidth=0.5, width=0.5)
    ax2.set_title(f"Correction\nΔ={correction_report.delta_pp:+.2f}pp "
                  f"(p={correction_report.p_value:.3f})")
    ax2.set_ylabel("Target Accuracy (%)"); ax2.grid(axis="y", alpha=0.3)

    # Panel 3: validation
    ax3 = axes[2]
    gaps = [validation_report.pre_gap, validation_report.post_gap]
    colors3 = ["#dc2626" if g > 0 else "#16a34a" for g in gaps]
    ax3.bar(["Pre-gap", "Post-gap"], gaps, color=colors3, alpha=0.85,
            edgecolor="black", linewidth=0.5, width=0.5)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_title(f"Validation\nCorrection magnitude="
                  f"{validation_report.correction_magnitude:+.2f}pp")
    ax3.set_ylabel("Performance Gap (pp)"); ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.show()
    return fig
