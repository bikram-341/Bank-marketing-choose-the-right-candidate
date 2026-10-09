"""
Unit tests for customer ranking and top 20% selection module.
"""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing import split_and_preprocess
from src.train_models import train_model
from src.customer_ranking import rank_and_select_customers, generate_export_dataframe


@pytest.fixture
def trained_pipeline():
    np.random.seed(42)
    n = 200
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
    prep_res = split_and_preprocess(df, random_state=42)
    model = train_model("decision_tree", prep_res["X_train_transformed"], prep_res["y_train"], random_state=42)
    return prep_res["preprocessor"], model


def test_customer_ranking_top_20_percent(trained_pipeline):
    preprocessor, model = trained_pipeline
    
    # 500 customers pool
    n_cust = 500
    np.random.seed(123)
    customers = pd.DataFrame({
        "age": np.random.randint(20, 70, size=n_cust),
        "job": np.random.choice(["admin.", "blue-collar", "technician", "retired"], size=n_cust),
        "marital": np.random.choice(["married", "single", "divorced"], size=n_cust),
        "education": np.random.choice(["secondary", "tertiary", "primary"], size=n_cust),
        "balance": np.random.randint(-500, 15000, size=n_cust),
        "housing": np.random.choice(["yes", "no"], size=n_cust),
        "loan": np.random.choice(["yes", "no"], size=n_cust),
        "contact": np.random.choice(["cellular", "telephone"], size=n_cust),
        "month": np.random.choice(["may", "jun", "jul", "aug"], size=n_cust),
        "campaign": np.random.randint(1, 10, size=n_cust),
        "pdays": np.random.choice([-1, 99, 180], size=n_cust),
        "previous": np.random.randint(0, 5, size=n_cust),
        "poutcome": np.random.choice(["unknown", "failure", "success"], size=n_cust),
    })

    ranked_df, stats = rank_and_select_customers(
        customers,
        preprocessor=preprocessor,
        model=model,
        top_pct=0.20,
    )

    # Validate ranking is descending
    probs = ranked_df["Predicted_Probability"].values
    assert (np.diff(probs) <= 1e-6).all(), "Probabilities must be sorted descending"

    # Validate exact 20% selected: 500 * 0.20 = 100
    selected_count = ranked_df["Selected_For_Campaign"].sum()
    assert selected_count == 100
    assert stats["selected_customers"] == 100
    assert stats["selection_percentage"] == 20.0
    assert ranked_df.iloc[0]["Rank"] == 1
    assert ranked_df.iloc[-1]["Rank"] == 500


def test_capacity_override_selection(trained_pipeline):
    preprocessor, model = trained_pipeline
    
    # Simulate bank scenario: 1000 customers, capacity constrained to exactly 200
    n_cust = 1000
    np.random.seed(99)
    customers = pd.DataFrame({
        "age": np.random.randint(20, 70, size=n_cust),
        "job": np.random.choice(["admin.", "blue-collar", "technician", "retired"], size=n_cust),
        "marital": np.random.choice(["married", "single", "divorced"], size=n_cust),
        "education": np.random.choice(["secondary", "tertiary", "primary"], size=n_cust),
        "balance": np.random.randint(-500, 15000, size=n_cust),
        "housing": np.random.choice(["yes", "no"], size=n_cust),
        "loan": np.random.choice(["yes", "no"], size=n_cust),
        "contact": np.random.choice(["cellular", "telephone"], size=n_cust),
        "month": np.random.choice(["may", "jun", "jul", "aug"], size=n_cust),
        "campaign": np.random.randint(1, 10, size=n_cust),
        "pdays": np.random.choice([-1, 99, 180], size=n_cust),
        "previous": np.random.randint(0, 5, size=n_cust),
        "poutcome": np.random.choice(["unknown", "failure", "success"], size=n_cust),
    })

    ranked_df, stats = rank_and_select_customers(
        customers,
        preprocessor=preprocessor,
        model=model,
        target_capacity=200,
    )

    assert stats["selected_customers"] == 200
    assert ranked_df["Selected_For_Campaign"].sum() == 200


def test_export_dataframe_structure(trained_pipeline):
    preprocessor, model = trained_pipeline
    n_cust = 50
    customers = pd.DataFrame({
        "age": [30] * n_cust,
        "job": ["admin."] * n_cust,
        "marital": ["single"] * n_cust,
        "education": ["secondary"] * n_cust,
        "balance": [1000] * n_cust,
        "housing": ["no"] * n_cust,
        "loan": ["no"] * n_cust,
        "contact": ["cellular"] * n_cust,
        "month": ["may"] * n_cust,
        "campaign": [1] * n_cust,
        "pdays": [-1] * n_cust,
        "previous": [0] * n_cust,
        "poutcome": ["unknown"] * n_cust,
    })

    ranked_df, _ = rank_and_select_customers(customers, preprocessor, model, top_pct=0.20)
    export_df = generate_export_dataframe(ranked_df, only_selected=True)

    assert len(export_df) == 10
    assert list(export_df.columns[:4]) == ["Rank", "Customer_ID", "Predicted_Probability", "Campaign_Action"]
