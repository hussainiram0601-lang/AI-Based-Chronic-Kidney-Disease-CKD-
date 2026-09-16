# AI-Based-Chronic-Kidney-Disease-CKD-
# AI-Based Chronic Kidney Disease (CKD) Risk Prediction Pipeline



This repository contains an end-to-end Machine Learning pipeline for predicting Chronic Kidney Disease (CKD) risk using patient clinical data. The pipeline handles dirty data cleaning, missing value imputation, categorical encoding, feature scaling, model selection, hyperparameter tuning, 5-fold cross-validation, and model serialization for production deployment.

---

## 📌 Project Overview

Chronic Kidney Disease (CKD) diagnosis relies on multiple clinical parameters. This project builds a production-ready machine learning pipeline that preprocesses raw clinical datasets, evaluates multiple classification algorithms, and selects the optimal model serialized with `joblib` for deployment.

---

## 📊 Dataset Overview

* **Source File**: `kidney_disease.csv`

* **Target Variable**: `classification` (`ckd`: 1, `notckd`: 0)


* **Data Split**: 80% Train, 20% Test (Stratified by target variable)



### Features Included

| Feature Type | Feature Names | Preprocessing Technique |
| --- | --- | --- |
| **Numerical** (14)

 | `age`, `bp`, `sg`, `al`, `su`, `bgr`, `bu`, `sc`, `sod`, `pot`, `hemo`, `pcv`, `wc`, `rc`<br> | `KNNImputer(n_neighbors=5, weights="distance")` + `StandardScaler()`<br> |
| **Categorical** (10)

 | `rbc`, `pc`, `pcc`, `ba`, `htn`, `dm`, `cad`, `appet`, `pe`, `ane`<br> | `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(drop="first")`<br> |

---

## 🛠️ Pipeline Architecture & Workflow

1. **Data Cleaning & Preprocessing**:
* Removed identifier column (`id`).


* Converted numerical fields stored as strings (`pcv`, `wc`, `rc`) into numeric types.


* Stripped whitespace and normalized string formatting across all categorical fields.


* Created a consolidated Scikit-Learn `ColumnTransformer` combining numerical and categorical pipelines.




2. **Model Training & Comparison**:
* **Logistic Regression**

* **Decision Tree**

* **Random Forest**

* **Support Vector Machine (SVM)**

* **K-Nearest Neighbors (KNN)**



3. **Hyperparameter Tuning**:
* Evaluated model configurations using `GridSearchCV` with 5-fold cross-validation optimizing for ROC-AUC.





---

## 📈 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | 0.9875

 | 1.0000

 | 0.9800

 | 0.9899

 | 0.9987

 |
| **SVM** | 0.9875

 | 1.0000

 | 0.9800

 | 0.9899

 | 0.9980

 |
| **Random Forest (Final Deployment)** | **0.9750**<br> | **0.9800**<br> | **0.9800**<br> | **0.9800**<br> | **0.9993**<br> |
| **Decision Tree** | 0.9625

 | 0.9796

 | 0.9600

 | 0.9697

 | 0.9633

 |
| **KNN** | 0.9500

 | 1.0000

 | 0.9200

 | 0.9583

 | 0.9900

 |

---

## 🚀 Final Production Model

The **Random Forest Classifier** pipeline was chosen for deployment due to high cross-validated stability and top ROC-AUC score ($0.9993$).

### Selected Hyperparameters

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42
)

```

### Model Artifacts Saved

* **Model Pipeline**: `ckd_random_forest_pipeline.joblib`

* **Metadata**: `ckd_model_metadata.joblib`


---

## 🖥️ How to Run

### 1. Prerequisites & Environment Setup

Ensure Python 3.x is installed along with the required libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib

```

### 2. Training the Pipeline

Place `kidney_disease.csv` in the root directory and execute the script:

```bash
python machine_learning_pipeline.py

```

### 3. Loading Model Artifact for Inference

```python
import joblib
import pandas as pd

# Load saved pipeline
pipeline = joblib.load("ckd_random_forest_pipeline.joblib")

# Predict on new unseen dataframe
predictions = pipeline.predict(new_data)
probabilities = pipeline.predict_proba(new_data)[:, 1]

```

---

## 📦 System Dependencies

* **Python**: `3.x`

* **scikit-learn**: Data preprocessing, imputer pipelines, model training, evaluation metrics


* **pandas & numpy**: Data manipulation and numeric operations


* **matplotlib & seaborn**: EDA and evaluation visualizations


* **joblib**: Model serialization
