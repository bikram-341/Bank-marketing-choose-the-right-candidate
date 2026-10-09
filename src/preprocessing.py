"""
Preprocessing Module for Bank Marketing Project.

Implements robust data cleaning, target separation, leakage prevention,
ColumnTransformer for scaling & one-hot encoding, and stratified 80:20 splitting.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# Features that cause data leakage for proactive marketing targeting:
# As described in UCI documentation: "duration: last contact duration, in seconds.
# This attribute highly affects the output target (e.g., if duration=0 then y='no').
# Yet, the duration is not known before a call is performed. Hence, if the intention is to have
# a realistic predictive model to SELECT customers to call, duration MUST be discarded."
POTENTIAL_LEAKAGE_FEATURES = ["duration"]


def identify_feature_types(
    X: pd.DataFrame,
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Dynamically identifies numerical and categorical feature columns.
    """
    if numerical_cols is not None and categorical_cols is not None:
        return numerical_cols, categorical_cols

    num_cols = []
    cat_cols = []

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)

    return num_cols, cat_cols


def clean_raw_data(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    target_col: str = "y",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw dataframe: removes duplicates, strips text whitespace,
    and reports cleaning statistics.
    """
    initial_rows = len(df)
    cleaned_df = df.copy()

    # Strip whitespaces from object/string columns
    for col in cleaned_df.select_dtypes(include=["object"]).columns:
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()

    # Drop duplicate records if requested
    duplicates_removed = 0
    if drop_duplicates:
        duplicates_removed = int(cleaned_df.duplicated().sum())
        if duplicates_removed > 0:
            cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

    cleaning_meta = {
        "initial_rows": initial_rows,
        "final_rows": len(cleaned_df),
        "duplicates_removed": duplicates_removed,
    }
    return cleaned_df, cleaning_meta


def build_preprocessor(
    numerical_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """
    Builds a ColumnTransformer that imputes and scales numerical features,
    and imputes and one-hot encodes categorical features.
    Configured with handle_unknown='ignore' to gracefully handle unseen test/prod values.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numerical_cols:
        transformers.append(("num", num_pipeline, numerical_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str = "y",
    exclude_leakage: bool = True,
    excluded_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
    """
    Separates X and y. Handles target binarization ('yes' -> 1, 'no' -> 0).
    Excludes data leakage features (such as 'duration') if requested.
    
    Returns:
        X: features DataFrame
        y: target Series (1/0) or None if target_col is absent
        excluded: list of features dropped
    """
    if excluded_features is None:
        excluded_features = POTENTIAL_LEAKAGE_FEATURES if exclude_leakage else []

    cols_to_drop = [c for c in excluded_features if c in df.columns]

    if target_col in df.columns:
        y_raw = df[target_col]
        # Map target to binary integer
        if pd.api.types.is_numeric_dtype(y_raw):
            y = y_raw.astype(int)
        else:
            y = y_raw.astype(str).str.lower().map({
                "yes": 1, "1": 1, "true": 1, "y": 1,
                "no": 0, "0": 0, "false": 0, "n": 0
            })
            # If any unmapped values, fill with 0 or raise
            if y.isnull().any():
                y = y.fillna(0).astype(int)
        
        # Ensure target is NOT in X
        cols_to_drop_with_target = list(set(cols_to_drop + [target_col]))
        X = df.drop(columns=cols_to_drop_with_target)
    else:
        y = None
        X = df.drop(columns=cols_to_drop, errors="ignore")

    return X, y, cols_to_drop


def split_and_preprocess(
    df: pd.DataFrame,
    target_col: str = "y",
    test_size: float = 0.20,
    random_state: int = 42,
    exclude_leakage: bool = True,
    drop_duplicates: bool = True,
) -> Dict[str, Any]:
    """
    Complete end-to-end preprocessing workflow:
    1. Clean duplicates and standardizes strings
    2. Separates X and y (target is 'y', binary 0/1)
    3. Stratified 80:20 split to handle class imbalance
    4. Fits ColumnTransformer ONLY on X_train (preventing data leakage)
    5. Transforms both X_train and X_test
    
    Returns:
        Dictionary containing raw and processed splits, preprocessor, and metadata.
    """
    cleaned_df, cleaning_meta = clean_raw_data(df, drop_duplicates=drop_duplicates, target_col=target_col)
    
    if target_col not in cleaned_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X, y, excluded_cols = prepare_features_and_target(
        cleaned_df, 
        target_col=target_col, 
        exclude_leakage=exclude_leakage
    )

    num_cols, cat_cols = identify_feature_types(X)
    
    # Stratified train-test split (80:20)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Build and fit preprocessor strictly on X_train
    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_train_transformed = preprocessor.fit_transform(X_train_raw)
    X_test_transformed = preprocessor.transform(X_test_raw)

    # Get transformed feature names
    feature_names = list(preprocessor.get_feature_names_out())

    return {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_transformed": X_train_transformed,
        "X_test_transformed": X_test_transformed,
        "preprocessor": preprocessor,
        "numerical_cols": num_cols,
        "categorical_cols": cat_cols,
        "feature_names": feature_names,
        "excluded_cols": excluded_cols,
        "cleaning_meta": cleaning_meta,
        "target_col": target_col,
        "random_state": random_state,
    }
