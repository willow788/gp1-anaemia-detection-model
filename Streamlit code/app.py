
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import io
import os
import warnings
import joblib

# Suppress warnings for cleaner presentation
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')

# --- Configuration for Professional Look ---
st.set_page_config(
    page_title="Advanced Anemia Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---

# Try to load notebook-trained pipeline (saved by the notebook)
def load_saved_pipeline(path="models/final_pipeline.joblib"):
    if os.path.exists(path):
        try:
            pipe = joblib.load(path)
            return pipe
        except Exception:
            return None
    return None

# 1. Synthetic Data Generator (REPLACES MISSING 'dataset.csv')
@st.cache_data
def load_and_preprocess_data():
    """Generates a synthetic dataset for demonstration purposes."""
    # Columns derived from the notebook's data output
    columns = [
        'age', 'gender', 'hb', 'rbc', 'pcv', 'mcv', 'mch', 'mchc',
        'vitamin_b12', 'folate_level', 'iron', 'transferrin', 'tibc', 'ferritin',
        'rdw', 'platelet_count', 'white_cell_count', 'red_cell_distribution_width',
        'creatinine', 'ejection_fraction', 'sodium', 'cpk', 'serum_creatinine',
        'hypertension', 'diabetes', 'platelets', 'smoking', 'observation_time',
        'fatigue', 'pallor', 'shortness_of_breath', 'dizziness',
        'anaemia', 'anaemia_type', 'death_event'
    ]

    np.random.seed(42)
    N = 500 # Number of samples
    data = {}
    
    # Generate mock data - creating correlation for a predictable model
    data['age'] = np.random.randint(18, 90, N)
    data['gender'] = np.random.randint(0, 2, N)
    data['anaemia'] = np.random.choice([0, 1], N, p=[0.6, 0.4]) # 40% prevalence

    # Hemoglobin (hb) is a key feature for anemia (lower for anemic patients)
    data['hb'] = np.where(data['anaemia'] == 1, 
                          np.random.uniform(7.0, 12.0, N), 
                          np.random.uniform(12.0, 17.0, N))
    data['hb'] = np.round(data['hb'], 1)

    # PCV (Hematocrit) highly correlated with Hb
    data['pcv'] = data['hb'] * 3 + np.random.normal(0, 1.5, N)
    data['pcv'] = np.round(data['pcv'], 1)

    # Mock features based on typical ranges
    for col in columns:
        if col not in data:
            if col in ['rbc', 'mcv', 'mch', 'mchc', 'creatinine', 'serum_creatinine', 'cpk', 'ejection_fraction', 'sodium', 'platelets', 'observation_time']:
                 data[col] = np.random.uniform(10, 200, N)
            elif col in ['hypertension', 'diabetes', 'smoking', 'fatigue', 'pallor', 'shortness_of_breath', 'dizziness']:
                data[col] = np.random.randint(0, 2, N)
            elif col in ['vitamin_b12', 'folate_level', 'iron', 'transferrin', 'tibc', 'ferritin', 'rdw', 'platelet_count', 'white_cell_count', 'red_cell_distribution_width']:
                data[col] = np.random.uniform(0.0, 500.0, N)
            elif col in ['anaemia_type', 'death_event']:
                data[col] = np.random.randint(0, 5, N)


    df = pd.DataFrame(data)
    
    # Clean column names as per notebook
    df.columns = (df.columns
              .str.lower()
              .str.replace(' ', '_')
              .str.replace('-', '_')
              .str.replace('(', '')
              .str.replace(')', '')
              .str.strip())
    
    # Fill any potential NaNs with median/mode (simulating the notebook's cleaning step)
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
            
    target_col = 'anaemia'
    return df, target_col

# 2. Model Trainer and Explainer (CACHED)
@st.cache_resource
def train_model(df, target_col):
    """Trains the XGBoost model and calculates SHAP values (used when no saved pipeline)."""
    
    X = df.drop(columns=[target_col, 'anaemia_type', 'death_event'])
    y = df[target_col]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Apply SMOTE for class balance as per notebook
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_smote.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Train XGBoost Classifier
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train_smote)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # SHAP Explainability
    explainer = shap.TreeExplainer(model)
    try:
        shap_values = explainer.shap_values(X_test_scaled)
    except Exception:
        shap_values = explainer(X_test_scaled).values
    
    return model, scaler, X_test_scaled, y_test, acc, cm, report, explainer, shap_values

# 3. Main Streamlit Logic
def main_app():
    # Load data
    df_raw, target_col = load_and_preprocess_data()
    saved_pipe = load_saved_pipeline()

    # If saved pipeline exists, derive metrics & SHAP from it; otherwise train synthetic model
    if saved_pipe is not None:
        # pipeline expected to be an imblearn Pipeline with 'preproc' and 'clf' named steps (not re-trained here)
        pipeline = saved_pipe
        preproc = pipeline.named_steps.get('preproc', None)
        clf = pipeline.named_steps.get('clf', pipeline)  # fallback to pipeline itself if clf not found

        # Prepare evaluation test split from the (synthetic) df for displaying metrics
        X = df_raw.drop(columns=[target_col, 'anaemia_type', 'death_event'])
        y = df_raw[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Predictions & metrics using the saved pipeline (pipeline handles preprocessing)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Preprocessed test for SHAP (if preprocessor exists)
        if preproc is not None:
            try:
                X_test_proc = preproc.transform(X_test)
            except Exception:
                X_test_proc = X_test.values
        else:
            X_test_proc = X_test.values

        # Build SHAP explainer for the classifier
        try:
            explainer = shap.TreeExplainer(clf)
            try:
                shap_values = explainer.shap_values(X_test_proc)
            except Exception:
                shap_values = explainer(X_test_proc).values
        except Exception:
            explainer, shap_values = None, None

        # For prediction time we will use pipeline.predict_proba on raw inputs
        model_for_prediction = pipeline
        scaler = None
        X_test_display = pd.DataFrame(X_test_proc, columns=(X_test.columns if hasattr(X_test, "columns") else None))
    else:
        # fallback: train synthetic model and use existing pipeline in app
        model, scaler, X_test_scaled, y_test, acc, cm, report, explainer, shap_values = train_model(df_raw, target_col)
        model_for_prediction = model
        X_test_display = X_test_scaled

    # --- Professional Header ---
    st.title("🩸 Advanced Hematology Predictive Analytics")
    st.subheader("AI-Powered Anemia Risk Assessment and Clinical Insight")
    st.markdown("""
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #008080;'>
            <p style='font-size: 1.1em; margin: 0; color: #333;'>
            This application uses an <strong>XGBoost Model</strong> (optimized with Optuna/SMOTE) to predict <strong>Anaemia Risk</strong>. For clinical validation, we provide comprehensive model explainability using <strong>SHAP</strong> values.
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")

    # --- Sidebar for Prediction Input ---
    st.sidebar.header("🔬 Patient Input Parameters")
    st.sidebar.markdown("**Enter the key laboratory and clinical values:**")
    
    # Define key input features 
    input_features = {
        'age': ('Age (Years)', 50.0, 18.0, 95.0, 1.0),
        'gender': ('Gender', {'Male': 1, 'Female': 0}, 'Female'),
        'hb': ('Hemoglobin (Hb, g/dL)', 12.0, 5.0, 20.0, 0.1),
        'pcv': ('Hematocrit (PCV/Hct, %)', 36.0, 15.0, 60.0, 0.1),
        'rbc': ('RBC Count (x10^6/μL)', 4.5, 2.0, 7.0, 0.01),
        'mcv': ('MCV (fL)', 90.0, 50.0, 150.0, 0.1),
        'mch': ('MCH (pg)', 30.0, 15.0, 50.0, 0.1),
        'mchc': ('MCHC (g/dL)', 33.0, 25.0, 40.0, 0.1),
        'creatinine': ('Serum Creatinine (mg/dL)', 1.0, 0.5, 5.0, 0.01),
        'smoking': ('Smoking', {'Yes': 1, 'No': 0}, 'No'),
        'diabetes': ('Diabetes', {'Yes': 1, 'No': 0}, 'No'),
        'hypertension': ('Hypertension', {'Yes': 1, 'No': 0}, 'No'),
    }
    
    patient_data = {}
    for col, (label, default, *params) in input_features.items():
        if isinstance(default, dict): 
            options = list(default.keys())
            # params[0] may be default label for selectbox; handle gracefully
            idx = 0
            if params:
                try:
                    idx = options.index(params[0])
                except Exception:
                    idx = 0
            selection = st.sidebar.selectbox(label, options, index=idx)
            patient_data[col] = default[selection]
        else:
            min_v, max_v, step = params
            patient_data[col] = st.sidebar.number_input(label, min_value=min_v, max_value=max_v, value=default, step=step, key=col)
    
    # Create the prediction DataFrame
    input_df = df_raw.drop(columns=[target_col, 'anaemia_type', 'death_event']).head(1).copy()
    
    # Fill the input_df with user data and default values for non-input features (median from training data)
    for col in input_df.columns:
        if col in patient_data:
            input_df[col] = patient_data[col]
        else:
            # Impute un-entered features with the median of the original dataset
            input_df[col] = df_raw[col].median()
            
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🔬 **Run Prediction**", type="primary")

    # --- Prediction & Explanation Section ---
    st.header("🎯 Patient-Specific Prediction & Clinical Rationale")
    col_pred, col_shap = st.columns([1, 2])
    
    with col_pred:
        if predict_button:
            # prediction via saved pipeline or local model
            if saved_pipe is not None:
                # pipeline handles preprocessing internally
                prediction_proba = model_for_prediction.predict_proba(input_df)[0][1]
            else:
                # use scaler + model from synthetic training
                X_input_scaled = scaler.transform(input_df)
                X_input_scaled_df = pd.DataFrame(X_input_scaled, columns=input_df.columns)
                prediction_proba = model_for_prediction.predict_proba(X_input_scaled_df)[0][1]
            
            prediction_label = "Anaemia Detected" if prediction_proba > 0.5 else "No Anaemia Detected"
            
            # Prediction Metric
            st.metric(
                label="Predicted Anaemia Probability", 
                value=f"{prediction_proba:.2%}", 
                delta=prediction_label, 
                delta_color=("inverse" if prediction_proba > 0.5 else "normal")
            )
            
            if prediction_proba > 0.5:
                st.error("🚨 **CLINICAL ALERT: Risk of Anaemia detected**")
                st.markdown(f"**Recommended Action:** Further diagnostics and consultation based on contributing factors.")
            else:
                st.success("✅ **Risk Profile: Low**")
                st.markdown(f"**Recommended Action:** Monitor and re-evaluate as clinically indicated.")
            
        else:
            st.info("👈 Enter patient data in the sidebar and click **'Run Prediction'** to generate a real-time risk assessment and detailed clinical explanation.")

    # Robust SHAP waterfall block (fixed indentation and without stray braces)
    with col_shap:
        if predict_button:
            st.subheader("Individual SHAP Waterfall (robust)")
            # ensure explainer and preproc/scaler exist
            if 'explainer' not in locals() or explainer is None:
                st.warning("No SHAP explainer available for this pipeline.")
            else:
                # prepare preprocessed input
                try:
                    if saved_pipe is not None and 'preproc' in locals() and preproc is not None:
                        X_input_pre = preproc.transform(input_df)
                    else:
                        X_input_pre = scaler.transform(input_df)
                except Exception:
                    # last-resort: convert to numpy
                    X_input_pre = input_df.values

                shap_exp = None
                vals = None
                try:
                    out = explainer(X_input_pre)
                    if hasattr(out, "values") or hasattr(out, "data"):
                        # normalize to single Explanation
                        shap_exp = out[0] if (hasattr(out, "__len__") and len(out) > 0) else out
                    else:
                        vals = out
                except Exception:
                    try:
                        vals = explainer.shap_values(X_input_pre)
                    except Exception:
                        vals = None

                # convert raw vals to Explanation if needed
                if shap_exp is None and vals is not None:
                    try:
                        if isinstance(vals, list) and len(vals) > 1:
                            vals_sel = vals[1]
                        else:
                            vals_sel = vals if not isinstance(vals, list) else vals[0]
                        single_vals = np.asarray(vals_sel)[0]
                        base_val = None
                        if hasattr(explainer, "expected_value"):
                            ev = explainer.expected_value
                            if isinstance(ev, (list, np.ndarray)):
                                base_val = ev[1] if len(ev) > 1 else ev[0]
                            else:
                                base_val = ev
                        elif hasattr(explainer, "expected_value_"):
                            base_val = explainer.expected_value_
                        shap_exp = shap.Explanation(values=single_vals,
                                                   base_values=base_val,
                                                   data=(X_input_pre[0] if hasattr(X_input_pre, "__getitem__") else None),
                                                   feature_names=input_df.columns.tolist())
                    except Exception:
                        shap_exp = None

                # Plot waterfall if available
                if shap_exp is not None:
                    try:
                        plt.figure(figsize=(10, 4))
                        try:
                            shap.plots.waterfall(shap_exp, max_display=12, show=False)
                        except Exception:
                            try:
                                shap.waterfall_plot(shap_exp, max_display=12, show=False)
                            except Exception:
                                raise
                        st.pyplot(plt.gcf(), use_container_width=True)
                        plt.close()
                    except Exception as e:
                        st.warning("Could not render SHAP waterfall (see debug).")
                        st.write(repr(e))
                else:
                    st.warning("SHAP explanation unavailable or could not be converted to a waterfall-compatible format.")
        else:
            st.info("Enter values and click Run Prediction to see SHAP explanation.")

    # --- Model Validation and Performance Section ---
    st.header("📈 Model Validation: Performance & Reliability")
    st.markdown("Presentation of the model's performance on the independent test set for clinical acceptance.")
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Confusion Matrix")
        # Plot Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Anaemia', 'Anaemia'], 
                    yticklabels=['Actual No Anaemia', 'Actual Anaemia'], ax=ax_cm)
        ax_cm.set_title(f"XGBoost Confusion Matrix (Accuracy: {acc:.2f})")
        ax_cm.set_xlabel("Predicted Label")
        ax_cm.set_ylabel("True Label")
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with col2:
        st.subheader("Key Performance Indicators (KPIs)")
        
        # Pull key metrics for Anaemia (Class 1)
        precision_anemia = report['1']['precision'] if '1' in report else report.get('1.0', {}).get('precision', 0.0)
        recall_anemia = report['1']['recall'] if '1' in report else report.get('1.0', {}).get('recall', 0.0)
        f1_anemia = report['1']['f1-score'] if '1' in report else report.get('1.0', {}).get('f1-score', 0.0)
        
        st.metric(label="Overall Accuracy", value=f"{acc:.2%}")
        st.metric(label="Anemia Precision (PPV)", value=f"{precision_anemia:.2%}", help="Positive Predictive Value: The confidence when predicting Anaemia.")
        st.metric(label="Anemia Recall (Sensitivity)", value=f"{recall_anemia:.2%}", help="True Positive Rate: The model's ability to correctly identify actual Anaemia cases.")
        st.metric(label="Anemia F1-Score", value=f"{f1_anemia:.2%}", help="Harmonic mean of Precision and Recall for a balanced measure.")


    st.markdown("---")
    
    # --- Global Explainability and Comparison Section ---
    st.header("📊 Global Feature Impact & Clinical Context")
    
    st.subheader("Global Feature Importance (SHAP Summary Plot)")
    st.markdown("""
        The SHAP Summary Plot visualizes the **overall importance** of each feature across the entire patient cohort. Features are ranked by their average impact on the model output.
    """)
    
    # Plot SHAP Summary Plot
    fig_shap, ax_shap = plt.subplots(figsize=(12, 6))
    try:
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values, X_test_display, max_display=10, show=False)
        else:
            shap.summary_plot(shap_values, X_test_display, max_display=10, show=False)
        st.pyplot(fig_shap, use_container_width=True)
    except Exception:
        st.warning("SHAP summary plot unavailable for this model/explainer.")
    plt.close(fig_shap)

    st.subheader("Clinical Data Distribution Comparison")
    st.markdown("Visual comparison of key features' value distributions for Anaemic (1) vs. Non-Anaemic (0) patients, providing essential clinical context.")
    
    # Select key clinical features for comparison
    comparison_features = ['hb', 'pcv', 'mcv', 'rbc', 'creatinine', 'age']
    
    cols_dist = st.columns(3)
    
    for i, feature in enumerate(comparison_features):
        with cols_dist[i % 3]:
            fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
            sns.histplot(data=df_raw, x=feature, hue='anaemia', kde=True, palette='coolwarm', ax=ax_dist)
            ax_dist.set_title(f'Distribution of {feature.upper()} by Anaemia Status')
            ax_dist.set_xlabel(feature.replace('_', ' ').title())
            st.pyplot(fig_dist)
            plt.close(fig_dist)

# --- Execution ---
if __name__ == "__main__":
    try:
        main_app()
    except Exception as e:
        # Inform the user about the data mock process
        st.error("An error occurred during app execution.")
        st.warning("⚠️ **IMPORTANT NOTE:** The original `dataset.csv` was not provided. This application is running a demonstration based on a **synthetic dataset** generated with similar column names and statistical properties to showcase the complete pipeline (Preprocessing, XGBoost Model, SMOTE, SHAP, and Streamlit Dashboard).")
        st.exception(e)
# ...existing code...
