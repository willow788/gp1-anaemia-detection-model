"""
Model explainability utilities using SHAP. 
"""
import numpy as np
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
from typing import Optional


def create_shap_explainer(model: xgb.XGBClassifier, 
                         X_train: Optional[np.ndarray] = None) -> shap. Explainer:
    """
    Create a SHAP explainer for the model.
    
    Args:
        model: Trained XGBoost model
        X_train:  Training data (optional, for TreeExplainer)
        
    Returns:
        SHAP explainer object
    """
    print("Creating SHAP explainer...")
    
    # Use TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(model)
    
    print("✓ SHAP explainer created")
    return explainer


def calculate_shap_values(explainer: shap.Explainer, 
                         X:  np.ndarray) -> shap. Explanation:
    """
    Calculate SHAP values for given data.
    
    Args:
        explainer: SHAP explainer object
        X: Data to explain
        
    Returns: 
        SHAP values
    """
    print(f"Calculating SHAP values for {X.shape[0]} samples...")
    shap_values = explainer(X)
    print("✓ SHAP values calculated")
    return shap_values


def plot_shap_summary(shap_values: shap.Explanation, 
                     feature_names: Optional[list] = None,
                     max_display: int = 20) -> None:
    """
    Plot SHAP summary plot. 
    
    Args:
        shap_values: SHAP values
        feature_names: List of feature names
        max_display: Maximum number of features to display
    """
    print("Generating SHAP summary plot...")
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, feature_names=feature_names, 
                     max_display=max_display, show=False)
    plt.title('SHAP Summary Plot - Feature Importance', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def plot_shap_waterfall(shap_values: shap.Explanation, 
                       index: int = 0,
                       max_display: int = 20) -> None:
    """
    Plot SHAP waterfall plot for a single prediction.
    
    Args:
        shap_values: SHAP values
        index: Index of the sample to explain
        max_display: Maximum number of features to display
    """
    print(f"Generating SHAP waterfall plot for sample {index}...")
    
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_values[index], max_display=max_display, show=False)
    plt.title(f'SHAP Waterfall Plot - Sample {index}', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def plot_shap_force(shap_values: shap.Explanation, 
                   index: int = 0) -> None:
    """
    Plot SHAP force plot for a single prediction.
    
    Args:
        shap_values:  SHAP values
        index:  Index of the sample to explain
    """
    print(f"Generating SHAP force plot for sample {index}...")
    
    shap.plots.force(shap_values[index], matplotlib=True)
    plt.title(f'SHAP Force Plot - Sample {index}', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_shap_dependence(shap_values: shap. Explanation,
                        feature_name: str,
                        X: np.ndarray,
                        feature_names: list) -> None:
    """
    Plot SHAP dependence plot for a specific feature.
    
    Args:
        shap_values: SHAP values
        feature_name: Name of the feature to plot
        X: Feature data
        feature_names: List of all feature names
    """
    print(f"Generating SHAP dependence plot for '{feature_name}'...")
    
    if feature_name not in feature_names:
        print(f"Error: Feature '{feature_name}' not found in feature_names")
        return
    
    feature_idx = feature_names.index(feature_name)
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature_idx, shap_values. values, X, 
                        feature_names=feature_names, show=False)
    plt.title(f'SHAP Dependence Plot - {feature_name}', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_shap_bar(shap_values: shap.Explanation,
                 max_display: int = 20) -> None:
    """
    Plot SHAP bar plot showing mean absolute SHAP values.
    
    Args:
        shap_values: SHAP values
        max_display: Maximum number of features to display
    """
    print("Generating SHAP bar plot...")
    
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title('SHAP Bar Plot - Mean Feature Importance', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def generate_shap_report(model:  xgb.XGBClassifier,
                        X_test: np.ndarray,
                        feature_names: Optional[list] = None,
                        sample_indices: Optional[list] = None) -> shap. Explanation:
    """
    Generate comprehensive SHAP analysis report.
    
    Args:
        model: Trained XGBoost model
        X_test: Test data
        feature_names: List of feature names
        sample_indices:  List of sample indices to explain in detail
        
    Returns: 
        SHAP values
    """
    print("\n" + "="*50)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("="*50 + "\n")
    
    # Create explainer
    explainer = create_shap_explainer(model)
    
    # Calculate SHAP values
    shap_values = calculate_shap_values(explainer, X_test)
    
    # Summary plot
    plot_shap_summary(shap_values, feature_names)
    
    # Bar plot
    plot_shap_bar(shap_values)
    
    # Individual explanations
    if sample_indices is None:
        sample_indices = [0, 1, 2]  # Default:  first 3 samples
    
    for idx in sample_indices:
        if idx < len(X_test):
            plot_shap_waterfall(shap_values, index=idx)
    
    print("\n" + "="*50)
    print("SHAP ANALYSIS COMPLETED")
    print("="*50 + "\n")
    
    return shap_values
