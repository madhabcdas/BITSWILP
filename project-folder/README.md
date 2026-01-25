# Breast Cancer Classification using Machine Learning  
End‑to‑End ML Modeling • Evaluation • Streamlit Deployment

---

## a. Problem Statement  
Breast cancer is one of the most common cancers affecting women worldwide. Early and accurate diagnosis plays a crucial role in improving survival rates.  
This project builds a complete end‑to‑end machine learning pipeline to classify breast tumors as Malignant (M) or Benign (B) using the Diagnostic Wisconsin Breast Cancer Dataset.

The workflow includes:
- Training six different ML classification models  
- Comparing their performance using multiple evaluation metrics  
- Building an interactive Streamlit web application  
- Deploying the app on Streamlit Community Cloud  

---

## b. Dataset Description  
**Dataset Name:** Diagnostic_Wisconsin_Breast_Cancer_Database.csv  
**Rows:** 569  
**Columns:** 32  

### Key Features
- ID – Unique identifier  
- Diagnosis – Target variable (M = Malignant, B = Benign)  
- 30 numerical features describing tumor characteristics  

### Target Variable  
- 0 → Benign  
- 1 → Malignant  

---

## c. Models Used & Evaluation Metrics  
- Logistic Regression  
- Decision Tree Classifier  
- K‑Nearest Neighbor (kNN)  
- Naive Bayes (Gaussian)  
- Random Forest  
- XGBoost  

Metrics:
- Accuracy  
- AUC  
- Precision  
- Recall  
- F1 Score  
- MCC  

---

## d. Streamlit App Features  
- Dataset upload (CSV)  
- Model selection dropdown  
- Evaluation metrics display  
- Confusion matrix heatmap  
- Classification report  

---

## e. Project Structure
