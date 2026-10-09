"""
Customer Ranking and Top 20% Selection Module for Bank Marketing Project.

Implements:
1. Probability scoring for any customer cohort (with or without target label).
2. Descending customer ranking by propensity to subscribe.
3. Top 20% selection or budget/capacity-constrained selection (e.g. 10,000 out of 50,000).
4. Business lift, efficiency metrics, and CSV export preparation.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


def rank_and_select_customers(
    customer_df: pd.DataFrame,
    preprocessor: Any,
    model: Any,
    top_pct: float = 0.20,
    target_capacity: Optional[int] = None,
    target_col: str = "y",
    id_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ranks customers by predicted subscription probability and selects the top decile/segment.
    
    Args:
        customer_df: Input dataframe representing eligible customer pool.
        preprocessor: Fitted ColumnTransformer.
        model: Trained classifier supporting predict_proba.
        top_pct: Percentage to select (default 0.20 for top 20%).
        target_capacity: Exact count override (e.g. exactly 10,000 if 50,000 customers).
        target_col: Name of target column if present in customer_df (will be excluded from scoring).
        id_col: Optional identifier column name; if None, generates 'Customer_ID'.
        
    Returns:
        ranked_df: Customer dataframe with probabilities, rank, and selection flag.
        selection_stats: Summary metrics regarding the selected cohort and marketing lift.
    """
    scored_df = customer_df.copy()

    # Assign synthetic ID if no explicit customer ID column exists
    if id_col is not None and id_col in scored_df.columns:
        customer_ids = scored_df[id_col].values
    else:
        customer_ids = [f"CUST_{i+1:06d}" for i in range(len(scored_df))]
        scored_df.insert(0, "Customer_ID", customer_ids)

    # Exclude target column and ID column from feature matrix
    cols_to_exclude = ["Customer_ID"]
    if target_col in scored_df.columns:
        cols_to_exclude.append(target_col)
    
    # Also exclude leakage features if present
    from src.preprocessing import POTENTIAL_LEAKAGE_FEATURES
    cols_to_exclude.extend([c for c in POTENTIAL_LEAKAGE_FEATURES if c in scored_df.columns])

    feature_df = scored_df.drop(columns=[c for c in set(cols_to_exclude) if c in scored_df.columns], errors="ignore")

    # Transform features using preprocessor
    X_transformed = preprocessor.transform(feature_df)

    # Predict subscription probabilities
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_transformed)
        if probas.shape[1] == 2:
            sub_probas = probas[:, 1]
        else:
            sub_probas = probas[:, 0]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_transformed)
        # Sigmoid calibration fallback
        sub_probas = 1.0 / (1.0 + np.exp(-scores))
    else:
        sub_probas = model.predict(X_transformed).astype(float)

    scored_df["Predicted_Probability"] = np.round(sub_probas, 4)

    # Rank customers in descending order of probability
    scored_df = scored_df.sort_values(by="Predicted_Probability", ascending=False).reset_index(drop=True)
    scored_df["Rank"] = np.arange(1, len(scored_df) + 1)
    scored_df["Percentile_Rank"] = np.round((scored_df["Rank"] / len(scored_df)) * 100, 2)

    total_eligible = len(scored_df)

    # Determine cutoff count
    if target_capacity is not None and target_capacity > 0:
        selection_cutoff = min(target_capacity, total_eligible)
    else:
        selection_cutoff = int(np.ceil(total_eligible * top_pct))

    scored_df["Selected_For_Campaign"] = scored_df["Rank"] <= selection_cutoff
    scored_df["Campaign_Action"] = scored_df["Selected_For_Campaign"].map({
        True: "CONTACT (Top Segment)",
        False: "DO NOT CONTACT"
    })

    # Cohort analysis
    selected_cohort = scored_df[scored_df["Selected_For_Campaign"]]
    unselected_cohort = scored_df[~scored_df["Selected_For_Campaign"]]

    overall_avg_prob = float(scored_df["Predicted_Probability"].mean())
    selected_avg_prob = float(selected_cohort["Predicted_Probability"].mean()) if not selected_cohort.empty else 0.0
    unselected_avg_prob = float(unselected_cohort["Predicted_Probability"].mean()) if not unselected_cohort.empty else 0.0

    lift_factor = (selected_avg_prob / overall_avg_prob) if overall_avg_prob > 0 else 1.0
    expected_conversions_selected = float(selected_cohort["Predicted_Probability"].sum())
    expected_conversions_total = float(scored_df["Predicted_Probability"].sum())

    selection_stats: Dict[str, Any] = {
        "total_customers": total_eligible,
        "selected_customers": int(selection_cutoff),
        "unselected_customers": int(total_eligible - selection_cutoff),
        "selection_percentage": round((selection_cutoff / total_eligible) * 100, 2) if total_eligible > 0 else 0,
        "overall_avg_probability": round(overall_avg_prob, 4),
        "selected_avg_probability": round(selected_avg_prob, 4),
        "unselected_avg_probability": round(unselected_avg_prob, 4),
        "lift_factor": round(lift_factor, 2),
        "expected_conversions_selected": int(np.round(expected_conversions_selected)),
        "expected_conversions_total": int(np.round(expected_conversions_total)),
        "capture_rate_pct": round((expected_conversions_selected / expected_conversions_total) * 100, 2) if expected_conversions_total > 0 else 0,
        "min_selected_probability": float(selected_cohort["Predicted_Probability"].min()) if not selected_cohort.empty else 0.0,
        "max_selected_probability": float(selected_cohort["Predicted_Probability"].max()) if not selected_cohort.empty else 0.0,
    }

    return scored_df, selection_stats


def generate_export_dataframe(ranked_df: pd.DataFrame, only_selected: bool = False) -> pd.DataFrame:
    """
    Prepares a clean, exportable dataframe for marketing operations.
    Places Customer_ID, Rank, Predicted_Probability, and Campaign_Action at the front.
    """
    df_out = ranked_df.copy()
    if only_selected:
        df_out = df_out[df_out["Selected_For_Campaign"]].copy()

    priority_cols = ["Rank", "Customer_ID", "Predicted_Probability", "Campaign_Action"]
    other_cols = [c for c in df_out.columns if c not in priority_cols and c != "Selected_For_Campaign"]
    
    return df_out[priority_cols + other_cols]
