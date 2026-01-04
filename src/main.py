"""
Main pipeline for anaemia detection model.
"""
import sys
import logging
import warnings
import numpy as np
from typing import Optional

# Import all modules
from src.config import (
    RANDOM_STATE, SUPPRESS_WARNINGS, LEAKAGE_GLOBALS,
    DATASET_PATH, BACKUP_PATH
)
from src.data_loader import (
    load_dataset, load_backup_dataset, display_dataset_summary
)
from src.data_preprocessing import (
    standardize_column_names, detect_target_column, encode_target,
    split_features_target, check_missing_values, split_train_test,
    create_preprocessing_pipeline, preprocess_data, get_class_distribution
)
from src.model_training import (
    detect_xgboost_device, apply_smote, train_xgboost_model,
    save_model, get_feature_importance
)
from src.hyperparameter_tuning import optimize_hyperparameters
from src.model_evaluation import (
    evaluate_model, plot_confusion_matrix, plot_target_distribution,
    plot_feature_importance, generate_evaluation_report
)
from src.explainability import generate_shap_report


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clear_leakage_globals():
    """Clear potentially leaked global variables."""
    logger.info("Clearing potential data leakage variables...")
    cleared = []
    for var in LEAKAGE_GLOBALS:
        if var in globals():
            del globals()[var]
            cleared.append(var)
    
    if cleared:
        logger.info(f"Cleared variables: {cleared}")
    else:
        logger.info("No leakage variables found")


def run_pipeline(use_backup: bool = True,
                optimize:  bool = False,
                n_trials: int = 50,
                apply_smote_resampling: bool = True,
                generate_shap:  bool = True,
                save_final_model: bool = True):
    """
    Run the complete anaemia detection pipeline.
    
    Args:
        use_backup:  Whether to use backup dataset
        optimize: Whether to perform hyperparameter optimization
        n_trials: Number of Optuna trials (if optimize=True)
        apply_smote_resampling: Whether to apply SMOTE
        generate_shap: Whether to generate SHAP analysis
        save_final_model: Whether to save the trained model
    """
    logger.info("="*60)
    logger.info("ANAEMIA DETECTION MODEL PIPELINE")
    logger.info("="*60)
    
    # Suppress warnings if configured
    if SUPPRESS_WARNINGS:
        warnings.filterwarnings('ignore')
        logger.info("Warnings suppressed")
    
    # Clear leakage variables
    clear_leakage_globals()
    
    # Step 1: Load Data
    logger.info("\n[STEP 1] Loading Dataset...")
    try:
        if use_backup:
            df = load_backup_dataset(BACKUP_PATH)
        else:
            df = load_dataset(DATASET_PATH)
        display_dataset_summary(df)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)
    
    # Step 2: Preprocessing
    logger.info("\n[STEP 2] Data Preprocessing...")
    
    # Standardize column names
    df = standardize_column_names(df)
    
    # Detect and encode target
    target_col = detect_target_column(df)
    df, label_encoder = encode_target(df, target_col)
    
    # Split features and target
    X, y = split_features_target(df, target_col)
    feature_names = X.columns.tolist()
    
    # Check missing values
    check_missing_values(df)
    
    # Show target distribution
    logger.info("\nTarget Distribution:")
    distribution = get_class_distribution(y)
    logger.info(f"Counts: {distribution['counts']}")
    logger.info(f"Imbalance Ratio: {distribution['imbalance_ratio']:.2f}")
    plot_target_distribution(y, title="Original Target Distribution")
    
    # Step 3: Train/Test Split
    logger.info("\n[STEP 3] Splitting Data...")
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    # Step 4:  Preprocessing Pipeline
    logger.info("\n[STEP 4] Creating Preprocessing Pipeline...")
    scaler = create_preprocessing_pipeline(X_train)
    X_train_proc = preprocess_data(X_train, scaler, fit=True)
    X_test_proc = preprocess_data(X_test, scaler, fit=False)
    
    # Step 5: Handle Imbalanced Data (SMOTE)
    if apply_smote_resampling:
        logger.info("\n[STEP 5] Applying SMOTE...")
        X_train_final, y_train_final = apply_smote(X_train_proc, y_train)
    else:
        logger.info("\n[STEP 5] Skipping SMOTE...")
        X_train_final, y_train_final = X_train_proc, np.array(y_train)
    
    # Step 6: Detect GPU/CPU
    logger.info("\n[STEP 6] Detecting Compute Device...")
    sample_size = min(100, len(X_train_final))
    tree_method = detect_xgboost_device(
        X_train_final[: sample_size],
        y_train_final[: sample_size]
    )
    
    # Step 7: Hyperparameter Optimization (Optional)
    best_params = None
    if optimize:
        logger.info(f"\n[STEP 7] Hyperparameter Optimization ({n_trials} trials)...")
        optim_results = optimize_hyperparameters(
            X_train_final, y_train_final,
            n_trials=n_trials,
            tree_method=tree_method,
            cv=5
        )
        best_params = optim_results['best_params']
        best_params['tree_method'] = tree_method
        best_params['random_state'] = RANDOM_STATE
    else:
        logger.info("\n[STEP 7] Skipping Hyperparameter Optimization...")
        best_params = {
            'tree_method': tree_method,
            'random_state': RANDOM_STATE,
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1
        }
    
    # Step 8: Train Final Model
    logger.info("\n[STEP 8] Training Final Model...")
    model = train_xgboost_model(X_train_final, y_train_final, params=best_params)
    
    # Step 9: Evaluate Model
    logger.info("\n[STEP 9] Evaluating Model...")
    class_names = ['No Death', 'Death'] if target_col == 'death_event' else None
    metrics = evaluate_model(model, X_test_proc, y_test, class_names)
    
    # Generate visualizations
    plot_confusion_matrix(metrics['confusion_matrix'], class_names)
    
    # Feature importance
    importance = get_feature_importance(model, feature_names)
    plot_feature_importance(importance, top_n=20)
    
    # Step 10: SHAP Explainability (Optional)
    if generate_shap:
        logger.info("\n[STEP 10] Generating SHAP Analysis...")
        try:
            shap_values = generate_shap_report(
                model, X_test_proc, 
                feature_names=feature_names,
                sample_indices=[0, 1, 2]
            )
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")
    else:
        logger.info("\n[STEP 10] Skipping SHAP Analysis...")
    
    # Step 11: Save Model
    if save_final_model:
        logger.info("\n[STEP 11] Saving Model...")
        model_path = save_model(model, 'anaemia_detection_model. pkl')
        logger.info(f"Model saved to:  {model_path}")
    else:
        logger.info("\n[STEP 11] Skipping Model Save...")
    
    # Pipeline Complete
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info(f"\nFinal Model Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Final Model F1 Score: {metrics['f1_score']:.4f}")
    
    return {
        'model':  model,
        'metrics': metrics,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'best_params': best_params
    }


if __name__ == "__main__": 
    """
    Run the pipeline with default settings.
    
    Modify parameters as needed:
    - use_backup: Use backup dataset instead of main dataset
    - optimize: Run Optuna hyperparameter optimization
    - n_trials: Number of optimization trials
    - apply_smote_resampling: Apply SMOTE for class balancing
    - generate_shap: Generate SHAP explainability analysis
    - save_final_model: Save the trained model
    """
    results = run_pipeline(
        use_backup=True,
        optimize=False,  # Set to True for hyperparameter tuning
        n_trials=50,
        apply_smote_resampling=True,
        generate_shap=True,
        save_final_model=True
    )
    
    logger.info("\nPipeline execution complete.  Results available in 'results' variable.")
