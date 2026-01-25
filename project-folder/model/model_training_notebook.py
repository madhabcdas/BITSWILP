# -------------------------------------------------------------
# Breast Cancer Classification – Training Script
# Diagnostic Wisconsin Breast Cancer Dataset
# -------------------------------------------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import joblib


# -------------------------------------------------------------
# Load Dataset
# -------------------------------------------------------------
df = pd.read_csv("data/Diagnostic_Wisconsin_Breast_Cancer_Database.csv")
print("\nDataset Loaded Successfully\n")
print(df.head())
df.columns = df.columns.str.replace(" ", "")


# -------------------------------------------------------------
# Encode Target + Drop ID Column
# -------------------------------------------------------------
df = df.copy()
le = LabelEncoder()
df["Diagnosis"] = le.fit_transform(df["Diagnosis"])  # M=1, B=0

df = df.drop(columns=["ID"])
print("\nAfter Encoding & Dropping ID Column:\n")
print(df.head())


# -------------------------------------------------------------
# Train-Test Split
# -------------------------------------------------------------
X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain-Test Split Completed\n")


# -------------------------------------------------------------
# Feature Scaling
# -------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature Scaling Completed\n")


# -------------------------------------------------------------
# Define Models
# -------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "kNN": KNeighborsClassifier(),
    "Naive Bayes (Gaussian)": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )
}

print("Models Initialized\n")


# -------------------------------------------------------------
# Evaluation Function
# -------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba):
    metrics = {}
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Precision"] = precision_score(y_true, y_pred)
    metrics["Recall"] = recall_score(y_true, y_pred)
    metrics["F1 Score"] = f1_score(y_true, y_pred)
    metrics["MCC"] = matthews_corrcoef(y_true, y_pred)

    try:
        metrics["AUC"] = roc_auc_score(y_true, y_proba[:, 1])
    except:
        metrics["AUC"] = np.nan

    return metrics


# -------------------------------------------------------------
# Train & Evaluate All Models
# -------------------------------------------------------------
results = {}
confusion_matrices = {}
classification_reports = {}

for name, model in models.items():

    print(f"\nTraining Model: {name}")

    if name in ["Logistic Regression", "kNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

    metrics = compute_metrics(y_test, y_pred, y_proba)
    results[name] = metrics

    confusion_matrices[name] = confusion_matrix(y_test, y_pred)
    classification_reports[name] = classification_report(y_test, y_pred)

print("\nAll Models Trained Successfully\n")


# -------------------------------------------------------------
# Display Results Table
# -------------------------------------------------------------
results_df = pd.DataFrame(results).T
print("\nModel Evaluation Metrics:\n")
print(results_df)


# -------------------------------------------------------------
# Save Trained Models
# -------------------------------------------------------------
print("\nSaving Models...\n")

for name, model in models.items():
    filename = f"model/{name.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
    joblib.dump(model, filename)

joblib.dump(scaler, "model/scaler.pkl")

print("All Models Saved Successfully\n")




# -------------------------------------------------------------
# Display Confusion Matrices
# -------------------------------------------------------------
for name, cm in confusion_matrices.items():
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix – {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# -------------------------------------------------------------
# Display Classification Reports
# -------------------------------------------------------------
for name, report in classification_reports.items():
    print(f"\n\n===== {name} =====\n")
    print(report)
