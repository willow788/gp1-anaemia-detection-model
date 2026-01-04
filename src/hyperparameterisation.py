"""
Hyperparameter tuning utilities using Optuna.
"""
import numpy as np
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from typing import Dict, Any
from src.config import RANDOM_STATE, OPTUNA_N_TRIALS, OPTUNA_SAMPLER


def create_optuna_objective(X_train: np.ndarray, y_train: np.ndarray, 
                            tree_method: str = 'hist', cv:  int = 5):
    """
    Create an Optuna objective function for hyperparameter optimization.
    
    Args:
        X_train: Training features
        y_train: Training labels
        tree_method: XGBoost tree method
        cv: Number of cross-validation folds
        
    Returns:
        Objective function for Optuna
    """
    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function to maximize cross-validated accuracy.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Mean cross-validation score
        """
        # Define hyperparameter search space
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial. suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'tree_method': tree_method,
            'random_state':  RANDOM_STATE,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        
        model = xgb.XGBClassifier(**params)
        
        # Cross-validation
        scores = cross_val_score(model, X_train, y_train, 
                                cv=cv, scoring='accuracy', n_jobs=-1)
        
        return scores.mean()
    
    return objective


def optimize_hyperparameters(X_train: np.ndarray, y_train: np.ndarray,
                             n_trials: int = OPTUNA_N_TRIALS,
                             tree_method: str = 'hist',
                             cv: int = 5,
                             show_progress: bool = True) -> Dict[str, Any]:
    """
    Optimize XGBoost hyperparameters using Optuna.
    
    Args:
        X_train:  Training features
        y_train:  Training labels
        n_trials:  Number of optimization trials
        tree_method: XGBoost tree method
        cv: Number of cross-validation folds
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary containing best parameters and study object
    """
    print(f"\nStarting hyperparameter optimization with Optuna...")
    print(f"Number of trials: {n_trials}")
    print(f"Cross-validation folds: {cv}")
    print(f"Tree method: {tree_method}")
    
    # Create objective function
    objective = create_optuna_objective(X_train, y_train, tree_method, cv)
    
    # Create study
    sampler = TPESampler(seed=RANDOM_STATE) if OPTUNA_SAMPLER == 'TPE' else None
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name='xgboost_optimization'
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, 
                   show_progress_bar=show_progress)
    
    # Results
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Best trial: {study.best_trial. number}")
    print(f"Best accuracy: {study.best_value:.4f}")
    print(f"\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*50 + "\n")
    
    return {
        'best_params':  study.best_params,
        'best_score': study.best_value,
        'best_trial': study.best_trial,
        'study': study
    }


def get_optimization_history(study: optuna.Study) -> Dict[str, list]:
    """
    Extract optimization history from Optuna study. 
    
    Args:
        study: Optuna study object
        
    Returns:
        Dictionary with trial numbers and values
    """
    trials = study.trials
    history = {
        'trial_numbers': [t.number for t in trials],
        'values': [t. value for t in trials],
        'params': [t. params for t in trials]
    }
    return history


def print_optimization_summary(study: optuna.Study, top_n: int = 5) -> None:
    """
    Print a summary of top performing trials.
    
    Args:
        study: Optuna study object
        top_n: Number of top trials to display
    """
    print(f"\nTop {top_n} Trials:")
    print("-" * 80)
    
    trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:top_n]
    
    for i, trial in enumerate(trials, 1):
        print(f"\n{i}. Trial #{trial.number}")
        print(f"   Score: {trial.value:.4f}")
        print(f"   Parameters:")
        for key, value in trial.params.items():
            print(f"     {key}: {value}")
    
    print("-" * 80)
