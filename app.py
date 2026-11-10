import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="BITS ML Assignment 2 - Model Evaluation App",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Machine Learning Model Evaluation Dashboard")
st.markdown("**BITS Pilani - Work Integrated Learning Programmes (WILP)**")
st.markdown("This app evaluates pre-trained classification models built on the Portuguese Student Achievement dataset (`student-por.csv`).")

# Sidebar for Model & Data Selection
st.sidebar.header("Configuration Panel")

# Model Selection mapping to your model/ folder
model_options = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/support_vector_machine.pkl"
}

selected_model_name = st.sidebar.selectbox("Select Classification Model", list(model_options.keys()))

# File Uploader for Test Data
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
        st.subheader("📁 Uploaded Test Dataset Preview")
        st.dataframe(test_df.head())
        
        if st.button("Run Evaluation"):
            model_path = model_options[selected_model_name]
            
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                st.success(f"Successfully loaded **{selected_model_name}**!")
                
                # Check if target column exists in uploaded test set
                if "target" in test_df.columns:
                    X_test = test_df.drop(columns=["target"])
                    y_test = test_df["target"]
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
                    
                    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
                    
                    acc = accuracy_score(y_test, y_pred)
                    auc = roc_auc_score(y_test, y_prob)
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    mcc = matthews_corrcoef(y_test, y_pred)
                    
                    # Display metrics
                    st.subheader(f"📈 Evaluation Metrics for {selected_model_name}")
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    col1.metric("Accuracy", round(acc, 4))
                    col2.metric("AUC", round(auc, 4))
                    col3.metric("Precision", round(prec, 4))
                    col4.metric("Recall", round(rec, 4))
                    col5.metric("F1 Score", round(f1, 4))
                    col6.metric("MCC", round(mcc, 4))
                else:
                    st.warning("Uploaded CSV must contain a 'target' column for evaluation metrics calculation.")
            else:
                st.error(f"Model file not found at {model_path}. Please check your model folder structure in GitHub.")
                
    except Exception as e:
        st.error(f"Error processing file or running model: {e}")
else:
    st.warning("⚠️ Please upload your test dataset CSV file using the sidebar to begin evaluation.")
    st.markdown("### Instructions:")
    st.markdown("1. Select your trained classification model from the sidebar dropdown.")
    st.markdown("2. Upload your test dataset CSV file.")
    st.markdown("3. Click **Run Evaluation** to view all performance metrics instantly.")
