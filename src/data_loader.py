"""
Data loading utilities for the anaemia detection pipeline. 
"""
import os
import pandas as pd
from typing import Tuple, Optional
from src.config import DATASET_PATH, BACKUP_PATH


def load_dataset(filepath: str = DATASET_PATH) -> pd.DataFrame:
    """
    Load the anaemia dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the dataset
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully from {filepath}")
        print(f"Shape: {df.shape}")
        return df
    except pd.errors. EmptyDataError:
        raise pd.errors.EmptyDataError(f"The file at {filepath} is empty")
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")


def load_backup_dataset(backup_path: str = BACKUP_PATH) -> pd.DataFrame:
    """
    Load the backup dataset (pre-leakage fix).
    
    Args:
        backup_path: Path to the backup CSV file
        
    Returns: 
        DataFrame containing the backup dataset
        
    Raises:
        FileNotFoundError: If backup file doesn't exist
    """
    if not os.path.exists(backup_path):
        raise FileNotFoundError(f"No backup found at {backup_path}")
    
    df = pd.read_csv(backup_path)
    print(f"Restored backup, df. shape = {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df


def validate_dataset(df: pd.DataFrame, required_columns: Optional[list] = None) -> bool:
    """
    Validate the dataset structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names (optional)
        
    Returns: 
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    if df.empty:
        raise ValueError("Dataset is empty")
    
    if required_columns: 
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Dataset validation passed.  Shape: {df.shape}")
    return True


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Get comprehensive information about the dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns: 
        Dictionary containing dataset statistics
    """
    info = {
        'shape': df.shape,
        'columns':  df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'total_missing': df.isnull().sum().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    return info


def display_dataset_summary(df: pd.DataFrame) -> None:
    """
    Display a summary of the dataset.
    
    Args:
        df:  DataFrame to summarize
    """
    info = get_dataset_info(df)
    
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Shape: {info['shape'][0]} rows × {info['shape'][1]} columns")
    print(f"Memory Usage: {info['memory_usage']:.2f} MB")
    print(f"Total Missing Values: {info['total_missing']}")
    
    if info['total_missing'] > 0:
        print("\nMissing Values per Column:")
        missing = {k: v for k, v in info['missing_values'].items() if v > 0}
        for col, count in missing.items():
            print(f"  {col}: {count}")
    
    print("\nFirst 5 rows:")
    print(df.head())
    print("="*50 + "\n")
