"""
Data Loading Module for Bank Marketing Project.

Responsible for discovering, loading, and summarizing the Bank Marketing dataset.
Supports both standard UCI (semicolon-separated) and Kaggle (comma-separated) formats.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import pandas as pd


# Default search locations for dataset
DEFAULT_DATA_DIRS = [
    Path("data"),
    Path("../data"),
    Path("."),
]

KNOWN_DATASET_NAMES = [
    "bank-full.csv",
    "bank.csv",
    "bank-additional-full.csv",
    "bank-additional.csv",
    "bank_marketing.csv",
]


class DatasetNotFoundError(FileNotFoundError):
    """Custom exception raised when bank marketing dataset is not located."""
    pass


def find_default_dataset_path(data_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """
    Search for known dataset filenames in data directories.
    
    Args:
        data_dir: Optional specific directory to search first.
        
    Returns:
        Path to found CSV file, or None if not found.
    """
    search_dirs = []
    if data_dir:
        search_dirs.append(Path(data_dir))
    search_dirs.extend(DEFAULT_DATA_DIRS)

    for directory in search_dirs:
        if not directory.exists() or not directory.is_dir():
            continue
        # Check known names first
        for name in KNOWN_DATASET_NAMES:
            candidate = directory / name
            if candidate.exists() and candidate.is_file():
                return candidate
        # Check any .csv with 'bank' in name
        for candidate in directory.glob("*bank*.csv"):
            if candidate.is_file():
                return candidate

    return None


def detect_delimiter(file_path_or_buffer: Union[str, Path, Any]) -> str:
    """
    Detect whether file is semicolon or comma separated.
    UCI bank marketing dataset is famously delimited by ';',
    while Kaggle versions are often delimited by ','.
    """
    try:
        if isinstance(file_path_or_buffer, (str, Path)):
            with open(file_path_or_buffer, "r", encoding="utf-8", errors="replace") as f:
                first_line = f.readline()
        else:
            # File-like object (e.g. Streamlit UploadedFile)
            pos = file_path_or_buffer.tell()
            first_line = file_path_or_buffer.readline()
            if isinstance(first_line, bytes):
                first_line = first_line.decode("utf-8", errors="replace")
            file_path_or_buffer.seek(pos)
        
        semicolon_count = first_line.count(";")
        comma_count = first_line.count(",")
        tab_count = first_line.count("\t")

        if semicolon_count > comma_count and semicolon_count > tab_count:
            return ";"
        elif tab_count > comma_count:
            return "\t"
        return ","
    except Exception:
        return ";"


def load_dataset(
    source: Optional[Union[str, Path, Any]] = None,
    target_col: str = "y"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads Bank Marketing Dataset from a file path or file-like buffer.
    
    Args:
        source: Path to CSV or file-like buffer (UploadedFile). If None, auto-discovers.
        target_col: Expected target column name (default 'y').
        
    Returns:
        Tuple of (DataFrame, summary_dict)
        
    Raises:
        DatasetNotFoundError: If dataset is not found or path is invalid.
    """
    file_path = None
    if source is None:
        file_path = find_default_dataset_path()
        if file_path is None:
            raise DatasetNotFoundError(
                "Bank Marketing Dataset not found.\n"
                "Please place 'bank-full.csv' or 'bank.csv' into the 'data/' folder,\n"
                "or run the download script: python scripts/download_dataset.py\n"
                "Dataset source: UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/222/bank+marketing)"
            )
        source_target = file_path
    else:
        source_target = source

    sep = detect_delimiter(source_target)
    
    try:
        df = pd.read_csv(source_target, sep=sep)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file with detected separator '{sep}': {e}")

    # Standardize column names (strip whitespace and quotes)
    df.columns = [c.strip().strip('"').strip("'") for c in df.columns]

    # Generate dataset summary
    summary = generate_dataset_summary(df, target_col=target_col, source_path=str(source_target))
    return df, summary


def generate_dataset_summary(
    df: pd.DataFrame, 
    target_col: str = "y", 
    source_path: str = ""
) -> Dict[str, Any]:
    """
    Generates comprehensive dataset diagnostics according to requirements.
    """
    missing_counts = df.isnull().sum().to_dict()
    total_missing = sum(missing_counts.values())
    duplicate_count = int(df.duplicated().sum())
    
    has_target = target_col in df.columns
    target_distribution = {}
    if has_target:
        target_distribution = df[target_col].value_counts(dropna=False).to_dict()

    summary: Dict[str, Any] = {
        "source": source_path,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "shape": df.shape,
        "column_names": list(df.columns),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "first_five": df.head(5),
        "missing_counts": missing_counts,
        "total_missing": total_missing,
        "duplicate_count": duplicate_count,
        "has_target": has_target,
        "target_col": target_col,
        "target_distribution": target_distribution,
    }
    return summary
