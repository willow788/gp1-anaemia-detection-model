"""
Data preprocessing utilities for the anaemia detection pipeline.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from src.config import TARGET_CANDIDATES, RANDOM_STATE, TEST_SIZE


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names:  lowercase, replace spaces and special characters. 
    
    Args:
        df: Input DataFrame
        
    Returns: 
        DataFrame with standardized column names
    """
    df_copy = df.copy()
    df_copy.columns = (
        df_copy.columns
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('-', '_')
        .str.replace('(', '')
        .str.replace(')', '')
        .str.strip()
    )
    print(f"Column names standardized: {df_copy.columns.tolist()}")
    return df_copy


def detect_target_column(df: pd.DataFrame, 
                         candidates: list = TARGET_CANDIDATES) -> str:
    """
    Detect the target column from a list of candidates.
    
    Args:
        df: Input DataFrame
        candidates: List of potential target column names (in priority order)
        
    Returns:
        Name of the detected target column
        
    Raises:
        ValueError: If no target column is found
    """
    for candidate in candidates:
        if candidate in df.columns:
            print(f"Target column detected: '{candidate}'")
            return candidate
    
    raise ValueError(f"No target column found.  Candidates: {candidates}")


def encode_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, Optional[LabelEncoder]]:
    """
    Encode target column if it's categorical.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        
    Returns: 
        Tuple of (DataFrame with encoded target, LabelEncoder or None)
    """
    df_copy = df.copy()
    
    if df_copy[target_col].dtype == 'object':
        le = LabelEncoder()
        df_copy[target_col] = le.fit_transform(df_copy[target_col])
        print(f"Encoded target column '{target_col}'")
        print(f"Classes: {le.classes_}")
        return df_copy, le
    
    print(f"Target column '{target_col}' is already numeric")
    return df_copy, None


def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataset into features (X) and target (y).
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]. copy()
    y = df[target_col].copy()
    
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"Number of features: {len(feature_cols)}")
    
    return X, y


def check_missing_values(df: pd.DataFrame, verbose: bool = True) -> pd.Series:
    """
    Check for missing values in the dataset.
    
    Args:
        df:  Input DataFrame
        verbose: Whether to print results
        
    Returns:
        Series of missing value counts per column
    """
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    
    if verbose:
        if len(missing_cols) == 0:
            print("✓ No missing values found")
        else:
            print("Missing values per column:")
            print(missing_cols)
            print(f"\nTotal missing values: {missing. sum()}")
    
    return missing


def create_preprocessing_pipeline(X_train: pd.DataFrame) -> StandardScaler:
    """
    Create and fit a preprocessing pipeline (StandardScaler).
    
    Args:
        X_train: Training features
        
    Returns:
        Fitted StandardScaler
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    print("Preprocessing pipeline (StandardScaler) created and fitted")
    return scaler


def preprocess_data(X: pd.DataFrame, scaler: StandardScaler, 
                    fit:  bool = False) -> np.ndarray:
    """
    Preprocess features using StandardScaler.
    
    Args:
        X: Features to preprocess
        scaler:  StandardScaler instance
        fit: Whether to fit the scaler (True for training data)
        
    Returns: 
        Preprocessed features as numpy array
    """
    if fit:
        X_processed = scaler.fit_transform(X)
        print(f"Fitted and transformed features.  Shape: {X_processed.shape}")
    else:
        X_processed = scaler.transform(X)
        print(f"Transformed features. Shape: {X_processed. shape}")
    
    return X_processed


def split_train_test(X: pd.DataFrame, y: pd.Series, 
                     test_size: float = TEST_SIZE,
                     random_state: int = RANDOM_STATE) -> Tuple: 
    """
    Split data into training and testing sets.
    
    Args:
        X:  Features
        y: Target
        test_size:  Proportion of test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain/Test Split (test_size={test_size}):")
    print(f"  X_train:  {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")
    
    # Show class distribution
    print(f"\nTraining set class distribution:")
    print(pd.Series(y_train).value_counts())
    print(f"\nTest set class distribution:")
    print(pd.Series(y_test).value_counts())
    
    return X_train, X_test, y_train, y_test


def get_class_distribution(y: pd.Series) -> dict:
    """
    Get the class distribution of the target variable.
    
    Args:
        y: Target variable
        
    Returns:
        Dictionary with class counts and percentages
    """
    counts = y.value_counts()
    percentages = y.value_counts(normalize=True) * 100
    
    distribution = {
        'counts': counts. to_dict(),
        'percentages': percentages.to_dict(),
        'imbalance_ratio': counts.max() / counts.min()
    }
    
    return distribution
