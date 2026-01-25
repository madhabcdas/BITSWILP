import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report
)
from pathlib import Path

st.set_page_config(page_title="Breast Cancer Classification", layout="wide")

st.markdown("""
<style>

    /* App background */
    [data-testid="stAppViewContainer"] {
        background-color: #f7f9fc;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #eef2f7;
    }

    /* Title */
    h1 {
        color: #1a365d !important;
        font-weight: 700 !important;
        text-align: center;
    }

    /* Subheaders */
    h2, h3 {
        color: #2b4c7e !important;
        font-weight: 600 !important;
    }

    /* File uploader label */
    [data-testid="stFileUploader"] label {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #1a365d !important;
    }

    /* Dataframe container */
    [data-testid="stDataFrame"] {
        border: 1px solid #d1d9e6 !important;
        border-radius: 10px !important;
        padding: 10px !important;
        background-color: white !important;
    }

    /* Buttons */
    button[kind="primary"] {
        background-color: #2b6cb0 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        border: none !important;
        transition: 0.3s !important;
    }

    button[kind="primary"]:hover {
        background-color: #1e4e79 !important;
        transform: scale(1.03);
    }

    /* Selectbox label */
    label[data-testid="stWidgetLabel"] {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #1a365d !important;
    }

    /* Alerts (error/info/warning) */
    .stAlert {
        border-radius: 10px !important;
        padding: 15px !important;
        font-size: 1rem !important;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        font-size: 0.9rem;
        color: #4a5568;
        margin-top: 30px;
    }

</style>
""", unsafe_allow_html=True)

st.title("🔬 Breast Cancer Classification - ML Model Evaluation App")
st.write("Upload test dataset, select a model, and view evaluation metrics.")

@st.cache_resource
# Get the directory where app.py is located
BASE_DIR = Path(__file__).resolve().parent
def load_models():
    # Construct absolute path: /mount/src/bitswilp/project-folder/model/Logistic_Regression.pkl
    model_path1 = BASE_DIR / "model" / "Logistic_Regression.pkl"
    model_path2 = BASE_DIR / "model" / "Decision_Tree.pkl"
    model_path3 = BASE_DIR / "model" / "kNN.pkl"
    model_path4 = BASE_DIR / "model" / "Naive_Bayes_Gaussian.pkl"
    model_path5 = BASE_DIR / "model" / "Random_Forest.pkl"
    model_path6 = BASE_DIR / "model" / "Random_Forest.pkl"

    models = {
        "Logistic Regression": model_path1,
        "Decision Tree Classifier": model_path2,
        "K-Nearest Neighbor Classifier": model_path3,
        "Naive Bayes Classifier": model_path4,
        "Ensemble Model - Random Forest": model_path5,
        "Ensemble Model - XGBoost": model_path6
    }

    scaler = joblib.load("model/scaler.pkl")
    return models, scaler

models, scaler = load_models()

st.subheader("📁 Upload Test Dataset (CSV Only)")
uploaded_file = st.file_uploader("Upload test CSV file", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(test_df.head())

    test_df.columns = test_df.columns.str.replace(" ", "")

    # Drop ID column if present
    if "ID" in test_df.columns:
        test_df = test_df.drop(columns=["ID"])

    # Check Diagnosis column
    if "Diagnosis" not in test_df.columns:
        st.error("❌ The uploaded file must contain a 'Diagnosis' column.")
        st.stop()

    # Required feature list
    required_features = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean",
        "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se",
        "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",
        "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
        "smoothness_worst", "compactness_worst", "concavity_worst",
        "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]

    # Validate schema
    uploaded_columns = list(test_df.columns)

    missing_cols = [col for col in required_features if col not in uploaded_columns]
    extra_cols = [col for col in uploaded_columns if col not in required_features + ["Diagnosis"]]

    # Show missing columns clearly
    if missing_cols:
        st.error("❌ The uploaded file is missing required feature columns.")
        st.warning("### Missing Columns:\n" + "\n".join([f"- {col}" for col in missing_cols]))
        st.stop()

    # Show extra columns (non-blocking)
    if extra_cols:
        st.info("### ℹ️ Extra columns detected (these will be ignored):\n" +
                "\n".join([f"- {col}" for col in extra_cols]))

    X_test = test_df.drop("Diagnosis", axis=1)
    y_test = test_df["Diagnosis"]

    if y_test.dtype == object:
        y_test = y_test.map({"B": 0, "M": 1})

    st.subheader("🤖 Select a Machine Learning Model")
    model_name = st.selectbox("Choose a model", list(models.keys()))
    model = models[model_name]

    if model_name in ["Logistic Regression", "K-Nearest Neighbor Classifier"]:
        X_input = scaler.transform(X_test)
    else:
        X_input = X_test

    y_pred = model.predict(X_input)
    y_proba = model.predict_proba(X_input)

    st.subheader("📊 Evaluation Metrics")

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "MCC", "AUC"],
        "Value": [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred),
            matthews_corrcoef(y_test, y_pred),
            roc_auc_score(y_test, y_proba[:, 1])
        ]
    })

    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("🧩 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.subheader("📄 Classification Report")
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    st.dataframe(report_df, use_container_width=True)

else:
    st.info("📌 Please upload a CSV file to begin.")

st.write("---")

st.markdown(
    "<p class='footer-text'>Developed by Madhab Chandra Das (2025ab05151@wilp.bits-pilani.ac.in)<br>"
    "</b>BITS Pilani WILP Machine Learning Lab Assignment</b></p>",
    unsafe_allow_html=True
)
