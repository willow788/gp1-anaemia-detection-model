"""
Model training utilities for the anaemia detection pipeline.
"""
import numpy as np
import xgboost as xgb
import joblib
import os
from typing import Tuple, Optional
from imblearn.over_sampling import SMOTE
from src.config import RANDOM_STATE, XGBOOST_PARAMS, MODEL_OUTPUT_DIR, SMOTE_RANDOM_STATE


def detect_xgboost_device(X_train_sample: np.ndarray, 
                          y_train_sample: np.ndarray) -> str:
    """
    Detect whether GPU is available for XGBoost.
    
    Args:
        X_train_sample: Small sample of training features for testing
        y_train_sample: Small sample of training labels for testing
        
    Returns:
        'gpu_hist' if GPU available, 'hist' otherwise
    """
    try:
        test_model = xgb.XGBClassifier(tree_method='gpu_hist', n_estimators=1)
        test_model.fit(X_train_sample, y_train_sample)
        print("✓ GPU detected and usable (gpu_hist)")
        return 'gpu_hist'
    except Exception as e:
        print(f"✗ No GPU or gpu_hist not available, using CPU (hist)")
        print(f"  Reason: {str(e)}")
        return 'hist'


def apply_smote(X_train: np.ndarray, y_train: np.ndarray, 
                random_state: int = SMOTE_RANDOM_STATE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to handle imbalanced data.
    
    Args:
        X_train:  Training features
        y_train:  Training labels
        random_state:  Random seed for reproducibility
        
    Returns:
        Tuple of (resampled X_train, resampled y_train)
    """
    print("\nApplying SMOTE for class balancing...")
    print(f"Before SMOTE - Class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} samples")
    
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"\nAfter SMOTE - Class distribution:")
    unique, counts = np.unique(y_resampled, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} samples")
    
    print(f"\nResampled data shape:")
    print(f"  X_resampled:  {X_resampled.shape}")
    print(f"  y_resampled: {y_resampled.shape}")
    
    return X_resampled, y_resampled


def train_xgboost_model(X_train: np. ndarray, y_train: np.ndarray,
                        params: Optional[dict] = None,
                        tree_method: str = 'hist') -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier. 
    
    Args:
        X_train: Training features
        y_train: Training labels
        params: Model parameters (uses XGBOOST_PARAMS if None)
        tree_method: Tree method ('hist' or 'gpu_hist')
        
    Returns:
        Trained XGBoost model
    """
    if params is None:
        params = XGBOOST_PARAMS. copy()
    
    params['tree_method'] = tree_method
    
    print(f"\nTraining XGBoost model with parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    print("✓ Model training completed")
    return model


def save_model(model: xgb.XGBClassifier, filename: str, 
               output_dir: str = MODEL_OUTPUT_DIR) -> str:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        filename: Name of the file (without path)
        output_dir: Directory to save the model
        
    Returns: 
        Full path to saved model
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    joblib.dump(model, filepath)
    print(f"✓ Model saved to:  {filepath}")
    
    return filepath


def load_model(filepath: str) -> xgb.XGBClassifier:
    """
    Load a trained model from disk. 
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded XGBoost model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found:  {filepath}")
    
    model = joblib.load(filepath)
    print(f"✓ Model loaded from: {filepath}")
    
    return model


def get_feature_importance(model: xgb.XGBClassifier, 
                          feature_names: Optional[list] = None) -> dict:
    """
    Get feature importance from trained model.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names (optional)
        
    Returns: 
        Dictionary mapping feature names to importance scores
    """
    importance_scores = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importance_scores))]
    
    importance_dict = dict(zip(feature_names, importance_scores))
    
    # Sort by importance
    importance_dict = dict(sorted(importance_dict.items(), 
                                  key=lambda x: x[1], reverse=True))
    
    print("\nTop 10 Most Important Features:")
    for i, (feature, score) in enumerate(list(importance_dict.items())[:10]):
        print(f"  {i+1}. {feature}: {score:.4f}")
    
    return importance_dict
