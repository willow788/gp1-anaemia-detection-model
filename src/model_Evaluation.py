"""
Model evaluation and visualization utilities.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score
)
from typing import Optional, Dict
import xgboost as xgb


def evaluate_model(model:  xgb.XGBClassifier, 
                  X_test: np.ndarray, 
                  y_test: np.ndarray,
                  class_names: Optional[list] = None) -> Dict:
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        class_names: List of class names for reporting
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model. predict_proba(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print("-" * 50)
    print(classification_report(y_test, y_pred, 
                               target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # ROC-AUC for binary classification
    if len(np.unique(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_pred_proba[: , 1])
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
    else:
        roc_auc = None
    
    print("="*50 + "\n")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'confusion_matrix': cm,
        'roc_auc': roc_auc
    }


def plot_confusion_matrix(cm: np.ndarray, 
                         class_names: Optional[list] = None,
                         figsize: tuple = (8, 6),
                         cmap: str = 'Blues') -> None:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        figsize:  Figure size
        cmap:  Colormap
    """
    plt.figure(figsize=figsize)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_target_distribution(y:  pd.Series, 
                            title: str = "Target Distribution",
                            class_names: Optional[list] = None,
                            figsize: tuple = (10, 6)) -> None:
    """
    Plot the distribution of the target variable.
    
    Args:
        y: Target variable
        title: Plot title
        class_names: List of class names
        figsize:  Figure size
    """
    plt.figure(figsize=figsize)
    
    counts = pd.Series(y).value_counts().sort_index()
    
    sns.barplot(x=counts.index, y=counts.values, palette='Set2')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # Add count labels on bars
    for i, v in enumerate(counts.values):
        plt.text(i, v + max(counts.values) * 0.01, str(v), 
                ha='center', va='bottom', fontweight='bold')
    
    # Set x-tick labels
    if class_names: 
        plt.xticks(range(len(class_names)), class_names)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\nTarget Distribution:")
    print(counts)
    print(f"\nTotal samples: {len(y)}")
    for idx, count in counts.items():
        percentage = (count / len(y)) * 100
        label = class_names[idx] if class_names and idx < len(class_names) else f"Class {idx}"
        print(f"{label}: {count} ({percentage:. 2f}%)")


def plot_roc_curve(y_test: np.ndarray, 
                   y_pred_proba: np.ndarray,
                   figsize: tuple = (8, 6)) -> None:
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
        figsize: Figure size
    """
    if len(np.unique(y_test)) != 2:
        print("ROC curve is only available for binary classification")
        return
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[: , 1])
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importance_dict: Dict[str, float],
                           top_n: int = 20,
                           figsize: tuple = (10, 8)) -> None:
    """
    Plot feature importance. 
    
    Args:
        importance_dict: Dictionary mapping feature names to importance scores
        top_n: Number of top features to plot
        figsize: Figure size
    """
    # Get top N features
    top_features = dict(list(importance_dict.items())[:top_n])
    
    plt.figure(figsize=figsize)
    
    features = list(top_features.keys())
    scores = list(top_features.values())
    
    # Create horizontal bar plot
    y_pos = np.arange(len(features))
    plt.barh(y_pos, scores, color='steelblue')
    
    plt.yticks(y_pos, features)
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Feature Importances', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.tight_layout()
    plt.show()


def generate_evaluation_report(model: xgb.XGBClassifier,
                              X_test: np.ndarray,
                              y_test: np. ndarray,
                              feature_names: Optional[list] = None,
                              class_names: Optional[list] = None) -> None:
    """
    Generate comprehensive evaluation report with visualizations.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        class_names: List of class names
    """
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, class_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'], class_names)
    
    # Plot ROC curve (binary only)
    if metrics['roc_auc'] is not None:
        plot_roc_curve(y_test, metrics['probabilities'])
    
    # Plot feature importance
    from src.model_training import get_feature_importance
    importance = get_feature_importance(model, feature_names)
    plot_feature_importance(importance)
