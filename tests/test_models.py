"""
Unit tests for model training and evaluation modules.
"""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing import split_and_preprocess
from src.train_models import train_model, build_svm_model, build_decision_tree_model
from src.evaluate_models import evaluate_single_model, compare_models, analyze_business_tradeoff


@pytest.fixture
def preprocessed_dataset():
    np.random.seed(42)
    n = 150
    df = pd.DataFrame({
        "age": np.random.randint(20, 70, size=n),
        "job": np.random.choice(["admin.", "blue-collar", "technician", "retired"], size=n),
        "marital": np.random.choice(["married", "single", "divorced"], size=n),
        "education": np.random.choice(["secondary", "tertiary", "primary"], size=n),
        "balance": np.random.randint(-500, 15000, size=n),
        "housing": np.random.choice(["yes", "no"], size=n),
        "loan": np.random.choice(["yes", "no"], size=n),
        "contact": np.random.choice(["cellular", "telephone"], size=n),
        "month": np.random.choice(["may", "jun", "jul", "aug"], size=n),
        "campaign": np.random.randint(1, 10, size=n),
        "pdays": np.random.choice([-1, 99, 180], size=n),
        "previous": np.random.randint(0, 5, size=n),
        "poutcome": np.random.choice(["unknown", "failure", "success"], size=n),
        "y": np.random.choice(["yes", "no"], size=n, p=[0.20, 0.80]),
    })
    return split_and_preprocess(df, test_size=0.20, random_state=42)


def test_svm_training_and_evaluation(preprocessed_dataset):
    X_train = preprocessed_dataset["X_train_transformed"]
    y_train = preprocessed_dataset["y_train"]
    X_test = preprocessed_dataset["X_test_transformed"]
    y_test = preprocessed_dataset["y_test"]

    svm = train_model("svm", X_train, y_train, random_state=42)
    eval_res = evaluate_single_model(svm, X_test, y_test, model_name="SVM")

    assert eval_res["accuracy"] >= 0.0 and eval_res["accuracy"] <= 1.0
    assert eval_res["precision"] >= 0.0 and eval_res["precision"] <= 1.0
    assert eval_res["recall"] >= 0.0 and eval_res["recall"] <= 1.0
    assert eval_res["f1_score"] >= 0.0 and eval_res["f1_score"] <= 1.0
    assert eval_res["confusion_matrix"].shape == (2, 2)
    assert eval_res["y_proba"] is not None
    assert (eval_res["y_proba"] >= 0.0).all() and (eval_res["y_proba"] <= 1.0).all()


def test_decision_tree_training_and_evaluation(preprocessed_dataset):
    X_train = preprocessed_dataset["X_train_transformed"]
    y_train = preprocessed_dataset["y_train"]
    X_test = preprocessed_dataset["X_test_transformed"]
    y_test = preprocessed_dataset["y_test"]

    tree = train_model("decision_tree", X_train, y_train, random_state=42)
    eval_res = evaluate_single_model(tree, X_test, y_test, model_name="Decision Tree")

    assert eval_res["accuracy"] >= 0.0 and eval_res["accuracy"] <= 1.0
    assert eval_res["confusion_matrix"].shape == (2, 2)
    assert eval_res["classification_report"] is not None


def test_model_comparison_and_business_impact():
    tradeoff = analyze_business_tradeoff(tn=80, fp=10, fn=5, tp=15, call_cost=5.0, deposit_profit=100.0)
    assert tradeoff["total_calls_made"] == 25
    assert tradeoff["wasted_marketing_cost"] == 50.0
    assert tradeoff["estimated_net_profit"] == (15 * 100.0) - (25 * 5.0)

    eval_mock = {
        "m1": {
            "model_name": "SVM", "accuracy": 0.85, "precision": 0.60,
            "recall": 0.70, "f1_score": 0.65, "roc_auc": 0.80,
            "tp": 14, "fp": 8, "fn": 6, "tn": 72
        }
    }
    comp_df = compare_models(eval_mock)
    assert len(comp_df) == 1
    assert comp_df.iloc[0]["Model"] == "SVM"
