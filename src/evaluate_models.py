"""
Model Evaluation Module for Bank Marketing Project.

Computes classification metrics, confusion matrices, ROC-AUC, classification reports,
and provides domain-specific business cost analysis (False Positives vs False Negatives).
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate_single_model(
    model: Any,
    X_test: np.ndarray,
    y_test: pd.Series,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier on the held-out test set.
    Calculates accuracy, precision, recall, f1, roc-auc, and confusion matrix.
    Safely handles edge cases if probabilities or certain metrics fail.
    """
    y_test_arr = np.array(y_test)
    y_pred = model.predict(X_test)

    # Calculate probabilities if supported
    y_proba = None
    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X_test)
            if probas.shape[1] == 2:
                y_proba = probas[:, 1]
                roc_auc = float(roc_auc_score(y_test_arr, y_proba))
            elif probas.shape[1] == 1:
                y_proba = probas[:, 0]
        except Exception:
            roc_auc = None
    elif hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X_test)
            roc_auc = float(roc_auc_score(y_test_arr, scores))
        except Exception:
            roc_auc = None

    # Calculate metrics
    accuracy = float(accuracy_score(y_test_arr, y_pred))
    precision = float(precision_score(y_test_arr, y_pred, zero_division=0))
    recall = float(recall_score(y_test_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_test_arr, y_pred, zero_division=0))

    cm = confusion_matrix(y_test_arr, y_pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    clf_report = classification_report(
        y_test_arr, y_pred,
        target_names=["No Deposit (0)", "Subscribed (1)"],
        output_dict=True,
        zero_division=0,
    )
    clf_report_text = classification_report(
        y_test_arr, y_pred,
        target_names=["No Deposit (0)", "Subscribed (1)"],
        zero_division=0,
    )

    business_impact = analyze_business_tradeoff(tn, fp, fn, tp)

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "classification_report": clf_report,
        "classification_report_text": clf_report_text,
        "business_impact": business_impact,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def analyze_business_tradeoff(
    tn: int, fp: int, fn: int, tp: int,
    call_cost: float = 5.0,
    deposit_profit: float = 120.0,
) -> Dict[str, Any]:
    """
    Analyzes False Positives vs False Negatives in a banking campaign context.
    
    False Positive (FP): Contacting a customer who will not subscribe.
      -> Cost: Operational call cost + sales time + mild customer annoyance.
    False Negative (FN): Missing a customer who was eager to subscribe.
      -> Cost: Severe opportunity loss (deposit margin & customer lifetime value).
    """
    total_calls_made = tp + fp
    estimated_marketing_cost = total_calls_made * call_cost
    estimated_gross_profit = tp * deposit_profit
    estimated_net_profit = estimated_gross_profit - estimated_marketing_cost

    lost_opportunity_cost = fn * deposit_profit
    wasted_marketing_cost = fp * call_cost

    explanation = (
        f"- False Positives ({fp:,}): Customers contacted who rejected the deposit, "
        f"costing approximately ${wasted_marketing_cost:,.2f} in wasted agent outreach.\n"
        f"- False Negatives ({fn:,}): Interested customers missed by the model, "
        f"resulting in approximately ${lost_opportunity_cost:,.2f} in lost deposit value.\n"
        f"In bank marketing, Recall and Top-decile ranking are critical because "
        f"the profit from a single term deposit (${deposit_profit}) vastly exceeds the marginal cost of a phone call (${call_cost})."
    )

    return {
        "call_cost": call_cost,
        "deposit_profit": deposit_profit,
        "total_calls_made": total_calls_made,
        "wasted_marketing_cost": wasted_marketing_cost,
        "lost_opportunity_cost": lost_opportunity_cost,
        "estimated_net_profit": estimated_net_profit,
        "explanation": explanation,
    }


def compare_models(evaluations: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Builds a structured comparison DataFrame from multiple model evaluations.
    """
    columns = [
        "Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC",
        "True Positives", "False Positives", "False Negatives", "True Negatives",
    ]
    if not evaluations:
        return pd.DataFrame(columns=columns)

    records = []
    for name, ev in evaluations.items():
        records.append({
            "Model": ev["model_name"],
            "Accuracy": round(ev["accuracy"], 4),
            "Precision": round(ev["precision"], 4),
            "Recall": round(ev["recall"], 4),
            "F1-Score": round(ev["f1_score"], 4),
            "ROC-AUC": round(ev["roc_auc"], 4) if ev["roc_auc"] is not None else np.nan,
            "True Positives": ev["tp"],
            "False Positives": ev["fp"],
            "False Negatives": ev["fn"],
            "True Negatives": ev["tn"],
        })
    return pd.DataFrame(records)
