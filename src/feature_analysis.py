"""
Feature Analysis Module for Bank Marketing Project.

Provides correlation analysis, Decision Tree feature importance calculation,
ranked importance tables, and business insights.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def compute_numerical_correlations(
    df: pd.DataFrame,
    target_col: str = "y",
) -> pd.DataFrame:
    """
    Computes Pearson correlation for numerical columns, including the binary target.
    """
    df_copy = df.copy()
    if target_col in df_copy.columns:
        if not pd.api.types.is_numeric_dtype(df_copy[target_col]):
            df_copy[target_col] = df_copy[target_col].astype(str).str.lower().map({
                "yes": 1, "1": 1, "true": 1, "y": 1,
                "no": 0, "0": 0, "false": 0, "n": 0
            }).fillna(0).astype(int)

    num_df = df_copy.select_dtypes(include=[np.number])
    if num_df.empty:
        return pd.DataFrame()

    corr_matrix = num_df.corr().round(4)
    return corr_matrix


def compute_tree_feature_importance(
    X_train_transformed: np.ndarray,
    y_train: pd.Series,
    feature_names: List[str],
    max_depth: int = 5,
    random_state: int = 42,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Trains an interpretable Decision Tree classifier on the preprocessed features
    and extracts ranked Gini feature importance.
    
    Returns:
        DataFrame with columns ['Rank', 'Feature', 'Importance', 'Percentage', 'Explanation']
    """
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=20,
        random_state=random_state,
        class_weight="balanced",
    )
    tree.fit(X_train_transformed, y_train)
    
    importances = tree.feature_importances_
    
    # Map back to feature names
    if len(feature_names) != X_train_transformed.shape[1]:
        # Fallback if names length mismatch
        feature_names = [f"Feature_{i}" for i in range(X_train_transformed.shape[1])]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    total_imp = importance_df["Importance"].sum()
    if total_imp > 0:
        importance_df["Percentage"] = (importance_df["Importance"] / total_imp) * 100
    else:
        importance_df["Percentage"] = 0.0

    importance_df["Rank"] = range(1, len(importance_df) + 1)
    
    # Add domain explanations
    importance_df["Explanation"] = importance_df["Feature"].apply(generate_feature_explanation)

    # Return top N
    return importance_df.head(top_n)


def generate_feature_explanation(feature_name: str) -> str:
    """
    Provides marketing domain explanation for key Bank Marketing features.
    """
    lower = feature_name.lower()
    if "poutcome" in lower:
        return "Outcome of previous marketing campaign (e.g. prior 'success' strongly boosts conversion likelihood)."
    elif "pdays" in lower:
        return "Number of days passed after previous contact (-1 means not previously contacted)."
    elif "previous" in lower:
        return "Number of contacts performed before this campaign for this client."
    elif "age" in lower:
        return "Client age (senior and younger segments historically exhibit higher subscription tendencies)."
    elif "balance" in lower:
        return "Average yearly balance (indicates liquidity and financial capacity to commit to term deposits)."
    elif "housing" in lower:
        return "Personal housing loan status (customers with existing mortgages are often less liquid)."
    elif "loan" in lower:
        return "Personal loan status (debt obligations hinder disposable savings commitments)."
    elif "contact" in lower:
        return "Communication contact type (cellular vs telephone/unknown contact efficiency)."
    elif "month" in lower:
        return "Contact month of the year (seasonal banking budget cycles e.g. March/September/October)."
    elif "campaign" in lower:
        return "Number of contacts performed during this campaign for this client."
    elif "education" in lower:
        return "Education level (tertiary education correlates with higher financial literacy)."
    elif "job" in lower:
        return "Occupation type (retirees, students, management exhibit distinct saving behaviors)."
    elif "marital" in lower:
        return "Marital status (influences household budgeting and financial stability)."
    else:
        return "Engineered demographic or transactional feature affecting subscription propensity."


def analyze_top_features(
    importance_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Produces executive summary insights from the top features.
    """
    top_3 = importance_df.head(3)["Feature"].tolist()
    total_top_3_pct = float(importance_df.head(3)["Percentage"].sum())
    
    return {
        "top_features": top_3,
        "top_3_coverage_pct": round(total_top_3_pct, 2),
        "summary_text": (
            f"The top predictive drivers are: {', '.join(top_3)}, "
            f"accounting for {total_top_3_pct:.1f}% of total decision influence. "
            "Prior relationship history and customer liquidity are primary determinants "
            "of term deposit conversion."
        ),
    }
