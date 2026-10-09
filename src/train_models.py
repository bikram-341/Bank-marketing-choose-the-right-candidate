"""
Model Training Module for Bank Marketing Project.

Implements:
1. Support Vector Machine (SVC with probability estimation, scaling, balanced weights)
2. Decision Tree Classifier (pruned with max_depth & min_samples_leaf, balanced weights)
3. Joblib model persistence and retrieval.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


MODELS_DIR = Path("models")


def build_svm_model(
    C: float = 1.0,
    kernel: str = "rbf",
    class_weight: str = "balanced",
    random_state: int = 42,
    cache_size: int = 500,
) -> SVC:
    """
    Constructs an SVC model suitable for imbalanced marketing classification
    with probability estimation enabled and standard convergence tolerance.
    """
    return SVC(
        C=C,
        kernel=kernel,
        probability=True,
        class_weight=class_weight,
        random_state=random_state,
        cache_size=cache_size,
    )


def build_decision_tree_model(
    max_depth: int = 6,
    min_samples_leaf: int = 20,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """
    Constructs a regularized Decision Tree Classifier to mitigate overfitting.
    """
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
    )


def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: pd.Series,
    random_state: int = 42,
    svm_max_samples: int = 6000,
) -> Any:
    """
    Trains either 'svm' or 'decision_tree'.
    For SVM on large datasets (> 6,000 samples), performs stratified subsampling
    during fit to ensure fast, high-quality convergence (< 6 seconds) on laptop hardware.
    """
    model_name_lower = model_name.lower().strip()
    
    if "svm" in model_name_lower or "svc" in model_name_lower:
        model = build_svm_model(random_state=random_state)
        # If dataset is very large, subsample stratify to maintain O(N^2) feasibility
        n_samples = X_train.shape[0]
        if n_samples > svm_max_samples:
            rng = np.random.RandomState(random_state)
            indices = np.arange(n_samples)
            # Stratified index sampling
            y_arr = np.array(y_train)
            pos_idx = indices[y_arr == 1]
            neg_idx = indices[y_arr == 0]
            
            n_pos = min(len(pos_idx), int(svm_max_samples * (len(pos_idx) / n_samples)))
            n_neg = min(len(neg_idx), svm_max_samples - n_pos)
            
            sampled_pos = rng.choice(pos_idx, size=n_pos, replace=False)
            sampled_neg = rng.choice(neg_idx, size=n_neg, replace=False)
            sampled_indices = np.concatenate([sampled_pos, sampled_neg])
            rng.shuffle(sampled_indices)
            
            X_fit = X_train[sampled_indices]
            y_fit = y_arr[sampled_indices]
        else:
            X_fit = X_train
            y_fit = y_train

        model.fit(X_fit, y_fit)
        return model

    elif "tree" in model_name_lower or "decision" in model_name_lower:
        model = build_decision_tree_model(random_state=random_state)
        model.fit(X_train, y_train)
        return model

    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose 'svm' or 'decision_tree'.")


def save_trained_artifacts(
    preprocessor: Any,
    models_dict: Dict[str, Any],
    metadata: Dict[str, Any],
    output_dir: Path = MODELS_DIR,
) -> Dict[str, str]:
    """
    Saves trained preprocessor, models, and metadata to disk using joblib.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    prep_path = output_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, prep_path)
    saved_paths["preprocessor"] = str(prep_path)

    for name, model in models_dict.items():
        clean_name = name.lower().replace(" ", "_")
        model_path = output_dir / f"{clean_name}_model.joblib"
        joblib.dump(model, model_path)
        saved_paths[clean_name] = str(model_path)

    meta_path = output_dir / "metadata.joblib"
    joblib.dump(metadata, meta_path)
    saved_paths["metadata"] = str(meta_path)

    return saved_paths


def load_trained_artifacts(
    models_dir: Path = MODELS_DIR,
) -> Tuple[Optional[Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Loads saved preprocessor, models, and metadata from disk.
    
    Returns:
        (preprocessor, models_dict, metadata)
    """
    if not models_dir.exists():
        return None, {}, None

    preprocessor = None
    prep_path = models_dir / "preprocessor.joblib"
    if prep_path.exists():
        preprocessor = joblib.load(prep_path)

    models_dict = {}
    for p in models_dir.glob("*_model.joblib"):
        name = p.stem.replace("_model", "")
        models_dict[name] = joblib.load(p)

    metadata = None
    meta_path = models_dir / "metadata.joblib"
    if meta_path.exists():
        metadata = joblib.load(meta_path)

    return preprocessor, models_dict, metadata
