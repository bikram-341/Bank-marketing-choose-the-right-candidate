"""
Visualization Module for Bank Marketing Project.

Produces clean, publication-ready figures using Matplotlib and Seaborn:
1. Target class distribution chart
2. Customer age distribution chart
3. Subscription rate by job category
4. Subscription rate by education and marital status
5. Confusion matrix heatmaps for both models
6. Model performance comparison chart
7. Feature importance chart
8. Customer subscription probability distribution
9. Top 20% selected vs remaining customer comparison
10. Cumulative gains / decile response chart
"""

from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# Set aesthetic styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "grid.alpha": 0.35,
    "figure.titlesize": 13,
})

COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "accent": "#2ca02c",
    "danger": "#d62728",
    "neutral": "#7f7f7f",
    "navy": "#0d3b66",
    "amber": "#f4d35e",
}


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = "y"
) -> plt.Figure:
    """1. Target class distribution chart."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    
    if target_col not in df.columns:
        ax.text(0.5, 0.5, f"Column '{target_col}' not found", ha="center")
        return fig

    counts = df[target_col].value_counts()
    percentages = (counts / len(df)) * 100

    bars = ax.bar(
        [str(k) for k in counts.index],
        counts.values,
        color=[COLORS["primary"], COLORS["accent"] if len(counts) > 1 else COLORS["primary"]],
        width=0.5,
        edgecolor="#333333",
        linewidth=0.8,
    )

    for bar, pct in zip(bars, percentages):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + (max(counts) * 0.02),
            f"{int(yval):,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Term Deposit Subscription Distribution (Target: 'y')", pad=15, fontweight="bold")
    ax.set_xlabel("Subscribed to Term Deposit")
    ax.set_ylabel("Number of Customers")
    ax.set_ylim(0, max(counts) * 1.2)
    fig.tight_layout()
    return fig


def plot_age_distribution(
    df: pd.DataFrame,
    target_col: str = "y"
) -> plt.Figure:
    """2. Customer age distribution with target overlay."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
    
    if "age" not in df.columns:
        ax.text(0.5, 0.5, "Age column not found", ha="center")
        return fig

    if target_col in df.columns:
        sns.histplot(
            data=df,
            x="age",
            hue=target_col,
            kde=True,
            bins=30,
            element="step",
            palette=[COLORS["primary"], COLORS["accent"]],
            ax=ax,
        )
        ax.legend(title="Subscribed", labels=["Yes", "No"], loc="upper right")
    else:
        sns.histplot(df["age"], kde=True, bins=30, color=COLORS["primary"], ax=ax)

    ax.set_title("Customer Age Distribution", pad=15, fontweight="bold")
    ax.set_xlabel("Customer Age (Years)")
    ax.set_ylabel("Customer Count")
    fig.tight_layout()
    return fig


def plot_subscription_by_job(
    df: pd.DataFrame,
    target_col: str = "y",
    top_n: int = 12,
) -> plt.Figure:
    """3. Subscription rate by job category."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=100)
    
    if "job" not in df.columns or target_col not in df.columns:
        ax.text(0.5, 0.5, "Required columns ('job' or target) missing", ha="center")
        return fig

    # Map target to binary
    df_temp = df.copy()
    if not pd.api.types.is_numeric_dtype(df_temp[target_col]):
        df_temp["y_bin"] = df_temp[target_col].astype(str).str.lower().isin(["yes", "1", "true"]).astype(int)
    else:
        df_temp["y_bin"] = df_temp[target_col].astype(int)

    job_rate = (
        df_temp.groupby("job")["y_bin"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "conversion_rate", "count": "total"})
        .sort_values(by="conversion_rate", ascending=True)
    )

    bars = ax.barh(
        job_rate.index,
        job_rate["conversion_rate"] * 100,
        color="#3a86ff",
        edgecolor="#222222",
        linewidth=0.6,
        height=0.6,
    )

    overall_avg = df_temp["y_bin"].mean() * 100
    ax.axvline(overall_avg, color="#e63946", linestyle="--", linewidth=1.5, label=f"Average ({overall_avg:.1f}%)")

    for bar, count in zip(bars, job_rate["total"]):
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}% (n={count:,})", va="center", fontsize=9)

    ax.set_title("Subscription Conversion Rate by Job Category (%)", pad=15, fontweight="bold")
    ax.set_xlabel("Subscription Rate (%)")
    ax.set_ylabel("Job Category")
    ax.set_xlim(0, max(job_rate["conversion_rate"] * 100) * 1.35)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_subscription_by_education_and_marital(
    df: pd.DataFrame,
    target_col: str = "y"
) -> plt.Figure:
    """4. Subscription rate by education and marital status."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=100)
    
    df_temp = df.copy()
    if target_col in df_temp.columns:
        if not pd.api.types.is_numeric_dtype(df_temp[target_col]):
            df_temp["y_bin"] = df_temp[target_col].astype(str).str.lower().isin(["yes", "1", "true"]).astype(int)
        else:
            df_temp["y_bin"] = df_temp[target_col].astype(int)
    else:
        ax1.text(0.5, 0.5, "Target column missing", ha="center")
        return fig

    # Education
    if "education" in df_temp.columns:
        edu_rate = (df_temp.groupby("education")["y_bin"].mean() * 100).sort_values(ascending=False)
        sns.barplot(x=edu_rate.index, y=edu_rate.values, hue=edu_rate.index, legend=False, ax=ax1, palette="Blues_r")
        ax1.set_title("Subscription Rate by Education", fontweight="bold")
        ax1.set_ylabel("Conversion Rate (%)")
        ax1.tick_params(axis="x", rotation=30)
        for i, v in enumerate(edu_rate.values):
            ax1.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

    # Marital
    if "marital" in df_temp.columns:
        mar_rate = (df_temp.groupby("marital")["y_bin"].mean() * 100).sort_values(ascending=False)
        sns.barplot(x=mar_rate.index, y=mar_rate.values, hue=mar_rate.index, legend=False, ax=ax2, palette="Greens_r")
        ax2.set_title("Subscription Rate by Marital Status", fontweight="bold")
        ax2.set_ylabel("Conversion Rate (%)")
        for i, v in enumerate(mar_rate.values):
            ax2.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    return fig


def plot_confusion_matrices(
    svm_cm: Optional[np.ndarray],
    tree_cm: Optional[np.ndarray],
) -> plt.Figure:
    """5. Confusion matrix heatmaps for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=100)
    labels = ["No Deposit", "Subscribed"]

    # SVM
    if svm_cm is not None:
        sns.heatmap(
            svm_cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[0],
            annot_kws={"size": 12, "weight": "bold"},
        )
        axes[0].set_title("SVM Confusion Matrix", fontweight="bold", pad=10)
        axes[0].set_xlabel("Predicted Label")
        axes[0].set_ylabel("Actual Label")
    else:
        axes[0].text(0.5, 0.5, "SVM not evaluated", ha="center")
        axes[0].axis("off")

    # Decision Tree
    if tree_cm is not None:
        sns.heatmap(
            tree_cm,
            annot=True,
            fmt="d",
            cmap="Greens",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[1],
            annot_kws={"size": 12, "weight": "bold"},
        )
        axes[1].set_title("Decision Tree Confusion Matrix", fontweight="bold", pad=10)
        axes[1].set_xlabel("Predicted Label")
        axes[1].set_ylabel("Actual Label")
    else:
        axes[1].text(0.5, 0.5, "Decision Tree not evaluated", ha="center")
        axes[1].axis("off")

    fig.tight_layout()
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame) -> plt.Figure:
    """6. Model performance comparison chart."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

    if comparison_df.empty or "Model" not in comparison_df.columns:
        ax.text(0.5, 0.5, "No model evaluation data available.\nPlease train models to view metric comparison.", ha="center", va="center", fontsize=11)
        ax.axis("off")
        fig.tight_layout()
        return fig

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    available_metrics = [m for m in metrics if m in comparison_df.columns]

    comp_copy = comparison_df.copy()
    for m in available_metrics:
        comp_copy[m] = pd.to_numeric(comp_copy[m], errors="coerce")

    valid_metrics = [m for m in available_metrics if not comp_copy[m].isna().all()]
    if not valid_metrics:
        ax.text(0.5, 0.5, "No numeric metrics available for plotting.", ha="center", va="center", fontsize=11)
        ax.axis("off")
        fig.tight_layout()
        return fig

    melted = comp_copy.melt(
        id_vars=["Model"],
        value_vars=valid_metrics,
        var_name="Metric",
        value_name="Score",
    ).dropna(subset=["Score"])

    if melted.empty:
        ax.text(0.5, 0.5, "No valid metric scores found.", ha="center", va="center", fontsize=11)
        ax.axis("off")
        fig.tight_layout()
        return fig

    sns.barplot(data=melted, x="Metric", y="Score", hue="Model", palette="Set2", ax=ax)
    ax.set_title("Model Evaluation Metrics Comparison", pad=15, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (0.0 to 1.0)")

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )

    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """7. Feature importance chart."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=100)
    
    top_df = importance_df.head(top_n).sort_values(by="Importance", ascending=True)

    bars = ax.barh(
        top_df["Feature"],
        top_df["Importance"],
        color="#4361ee",
        edgecolor="#1d3557",
        linewidth=0.6,
        height=0.6,
    )

    for bar, pct in zip(bars, top_df["Percentage"]):
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va="center", fontsize=9, fontweight="bold")

    ax.set_title(f"Top {top_n} Most Influential Features (Decision Tree)", pad=15, fontweight="bold")
    ax.set_xlabel("Relative Feature Importance (Gini)")
    ax.set_xlim(0, max(top_df["Importance"]) * 1.25)
    fig.tight_layout()
    return fig


def plot_probability_distribution(
    ranked_df: pd.DataFrame,
    threshold: Optional[float] = None,
) -> plt.Figure:
    """8. Customer subscription probability distribution."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
    
    if "Predicted_Probability" not in ranked_df.columns:
        ax.text(0.5, 0.5, "Probabilities not found", ha="center")
        return fig

    sns.histplot(
        ranked_df["Predicted_Probability"],
        bins=30,
        kde=True,
        color="#3f37c9",
        ax=ax,
    )

    if threshold is not None:
        ax.axvline(threshold, color="#d90429", linestyle="--", linewidth=1.8, label=f"Top 20% Cutoff ({threshold:.3f})")
        ax.legend(loc="upper right")

    ax.set_title("Customer Subscription Probability Distribution", pad=15, fontweight="bold")
    ax.set_xlabel("Predicted Probability of Subscription")
    ax.set_ylabel("Customer Count")
    fig.tight_layout()
    return fig


def plot_selected_vs_remaining(selection_stats: Dict[str, Any]) -> plt.Figure:
    """9. Top 20% selected versus remaining customers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=100)

    # Donut chart: Customer Allocation
    selected_n = selection_stats.get("selected_customers", 10000)
    unselected_n = selection_stats.get("unselected_customers", 40000)
    
    wedges, texts, autotexts = ax1.pie(
        [selected_n, unselected_n],
        labels=["Selected (Top 20%)", "Remaining (80%)"],
        autopct="%1.1f%%",
        colors=["#06d6a0", "#118ab2"],
        startangle=90,
        wedgeprops=dict(width=0.4, edgecolor="w", linewidth=2),
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax1.set_title(f"Customer Population Allocation\n(Total: {selected_n + unselected_n:,})", fontweight="bold")

    # Bar chart: Average Conversion Probability Comparison
    sel_p = selection_stats.get("selected_avg_probability", 0) * 100
    unsel_p = selection_stats.get("unselected_avg_probability", 0) * 100
    overall_p = selection_stats.get("overall_avg_probability", 0) * 100

    categories = ["Selected (Top 20%)", "Overall Population", "Remaining (80%)"]
    vals = [sel_p, overall_p, unsel_p]
    colors = ["#06d6a0", "#ffd166", "#118ab2"]

    bars = ax2.bar(categories, vals, color=colors, width=0.55, edgecolor="#222", linewidth=0.7)
    for b in bars:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 0.5, f"{h:.1f}%", ha="center", fontweight="bold", fontsize=10)

    ax2.set_ylabel("Avg Subscription Probability (%)")
    ax2.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 100)
    ax2.set_title(f"Targeting Efficiency (Lift: {selection_stats.get('lift_factor', 1.0)}x)", fontweight="bold")
    ax2.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    return fig


def plot_cumulative_gains(ranked_df: pd.DataFrame) -> plt.Figure:
    """10. Cumulative gains / decile response chart."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

    if "Predicted_Probability" not in ranked_df.columns:
        ax.text(0.5, 0.5, "Ranking data unavailable", ha="center")
        return fig

    # Compute cumulative probability response curve
    df_sorted = ranked_df.sort_values(by="Predicted_Probability", ascending=False).reset_index(drop=True)
    cum_expected = df_sorted["Predicted_Probability"].cumsum()
    total_expected = cum_expected.iloc[-1] if not cum_expected.empty else 1.0

    pct_customers = np.linspace(0, 100, len(df_sorted))
    pct_gains = (cum_expected / total_expected) * 100

    ax.plot(pct_customers, pct_gains, color="#06d6a0", linewidth=2.5, label="Model Selection Curve")
    ax.plot([0, 100], [0, 100], color="#999999", linestyle="--", linewidth=1.5, label="Random Contacting Baseline")

    # Mark 20% point
    idx_20 = int(len(df_sorted) * 0.20)
    gain_at_20 = pct_gains.iloc[idx_20] if idx_20 < len(pct_gains) else 0.0

    ax.plot(20, gain_at_20, marker="o", markersize=8, color="#ef476f")
    ax.axvline(20, color="#ef476f", linestyle=":", alpha=0.7)
    ax.axhline(gain_at_20, color="#ef476f", linestyle=":", alpha=0.7)
    ax.annotate(
        f"Top 20% captures\n{gain_at_20:.1f}% of conversions",
        xy=(20, gain_at_20),
        xytext=(28, gain_at_20 - 15),
        arrowprops=dict(arrowstyle="->", color="#ef476f", lw=1.5),
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff", ec="#ef476f", lw=1),
    )

    ax.set_title("Cumulative Gains Chart (Marketing Lift)", pad=15, fontweight="bold")
    ax.set_xlabel("% of Total Customers Contacted")
    ax.set_ylabel("% of Total Conversions Captured")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig
