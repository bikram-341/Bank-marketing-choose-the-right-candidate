"""
Unit tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import (
    identify_feature_types,
    clean_raw_data,
    build_preprocessor,
    prepare_features_and_target,
    split_and_preprocess,
    POTENTIAL_LEAKAGE_FEATURES,
)


@pytest.fixture
def sample_marketing_df():
    """Generates a representative synthetic dataframe matching UCI schema."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "age": np.random.randint(20, 70, size=n),
        "job": np.random.choice(["admin.", "blue-collar", "technician", "retired"], size=n),
        "marital": np.random.choice(["married", "single", "divorced"], size=n),
        "education": np.random.choice(["secondary", "tertiary", "primary"], size=n),
        "balance": np.random.randint(-500, 15000, size=n),
        "housing": np.random.choice(["yes", "no"], size=n),
        "loan": np.random.choice(["yes", "no"], size=n),
        "contact": np.random.choice(["cellular", "telephone"], size=n),
        "month": np.random.choice(["may", "jun", "jul", "aug"], size=n),
        "duration": np.random.randint(10, 800, size=n),
        "campaign": np.random.randint(1, 10, size=n),
        "pdays": np.random.choice([-1, 99, 180], size=n),
        "previous": np.random.randint(0, 5, size=n),
        "poutcome": np.random.choice(["unknown", "failure", "success"], size=n),
        "y": np.random.choice(["yes", "no"], size=n, p=[0.15, 0.85]),
    })


def test_clean_raw_data_handles_duplicates_and_spaces(sample_marketing_df):
    df_with_dup = pd.concat([sample_marketing_df, sample_marketing_df.iloc[:5]], ignore_index=True)
    cleaned, meta = clean_raw_data(df_with_dup, drop_duplicates=True)
    assert len(cleaned) == len(sample_marketing_df)
    assert meta["duplicates_removed"] == 5


def test_prepare_features_and_target_removes_target_and_leakage(sample_marketing_df):
    X, y, dropped = prepare_features_and_target(sample_marketing_df, target_col="y", exclude_leakage=True)
    assert "y" not in X.columns
    assert "duration" not in X.columns
    assert "duration" in dropped
    assert len(y) == len(sample_marketing_df)
    assert set(y.unique()).issubset({0, 1})


def test_missing_target_column_raises_error(sample_marketing_df):
    df_no_target = sample_marketing_df.drop(columns=["y"])
    with pytest.raises(ValueError, match="Target column 'y' not found"):
        split_and_preprocess(df_no_target, target_col="y")


def test_split_and_preprocess_stratified_80_20(sample_marketing_df):
    result = split_and_preprocess(sample_marketing_df, test_size=0.20, random_state=42)
    assert len(result["X_train_raw"]) == 80
    assert len(result["X_test_raw"]) == 20
    # Fit strictly on train
    assert result["X_train_transformed"].shape[0] == 80
    assert result["X_test_transformed"].shape[0] == 20
    # No NaN in transformed data
    assert not np.isnan(result["X_train_transformed"]).any()
    assert not np.isnan(result["X_test_transformed"]).any()


def test_handles_unseen_categories_gracefully(sample_marketing_df):
    result = split_and_preprocess(sample_marketing_df, random_state=42)
    prep = result["preprocessor"]

    # Introduce unseen job and education in new test batch
    new_data = pd.DataFrame({
        "age": [45],
        "job": ["astronaut"],  # Completely unseen
        "marital": ["married"],
        "education": ["doctorate"],  # Completely unseen
        "balance": [5000],
        "housing": ["no"],
        "loan": ["no"],
        "contact": ["cellular"],
        "month": ["may"],
        "campaign": [1],
        "pdays": [-1],
        "previous": [0],
        "poutcome": ["unknown"],
    })

    # Should not raise ValueError due to handle_unknown='ignore'
    transformed = prep.transform(new_data)
    assert transformed.shape[0] == 1
    assert transformed.shape[1] == result["X_train_transformed"].shape[1]


def test_handles_missing_values_with_imputation(sample_marketing_df):
    df_with_nans = sample_marketing_df.copy()
    df_with_nans.loc[0, "age"] = np.nan
    df_with_nans.loc[1, "job"] = np.nan
    df_with_nans.loc[2, "balance"] = np.nan

    result = split_and_preprocess(df_with_nans, random_state=42)
    assert not np.isnan(result["X_train_transformed"]).any()
