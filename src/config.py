"""
Configuration and constants for the anaemia detection pipeline.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# File paths
DATA_DIR = '../data'
DATASET_PATH = f'{DATA_DIR}/dataset.csv'
BACKUP_PATH = f'{DATA_DIR}/dataset_backup_pre_leakfix.csv'
MODEL_OUTPUT_DIR = '../models'

# Target column candidates (in priority order)
TARGET_CANDIDATES = ['death_event', 'anaemia_type', 'anaemia', 'diagnosis', 'target', 'label']

# Columns to clear for preventing data leakage
LEAKAGE_GLOBALS = [
    'preprocessor',
    'X_train_proc',
    'X_test_proc',
    'X_train_res',
    'y_train_res',
    'y_pred',
    'y_test_final',
    'final_model_booster'
]

# Visualization settings
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
COLOR_PALETTE = 'husl'
FIGURE_SIZE = (12, 6)
FONT_SIZE = 10

# Apply visualization settings
plt.style.use(PLOT_STYLE)
sns.set_palette(COLOR_PALETTE)
plt.rcParams['figure.figsize'] = FIGURE_SIZE
plt.rcParams['font.size'] = FONT_SIZE

# Model parameters
XGBOOST_PARAMS = {
    'random_state': RANDOM_STATE,
    'tree_method': 'hist',  # Will be updated based on GPU availability
    'eval_metric': 'logloss'
}

# Train/test split ratio
TEST_SIZE = 0.2

# SMOTE parameters
SMOTE_RANDOM_STATE = RANDOM_STATE

# Optuna parameters
OPTUNA_N_TRIALS = 100
OPTUNA_SAMPLER = 'TPE'  # Tree-structured Parzen Estimator

# Warning suppression
SUPPRESS_WARNINGS = True

print("Configuration loaded successfully!")
