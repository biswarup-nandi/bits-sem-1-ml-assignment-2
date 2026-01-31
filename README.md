# Machine Learning Assignment 2 — 6 Classifiers + Streamlit App (Breast Cancer)

**Author:** Biswarup Nandi  
**Student ID:** 2025AA05115  
**Email:** 2025aa05115@wilp.bits-pilani.ac.in  

---

## 1) Problem statement

Build and compare **six classification models** to predict whether a breast tumor is **malignant or benign**, and provide a simple **Streamlit UI** where a user can upload a CSV and view predictions + evaluation metrics.

---

## 2) Dataset description

- **Dataset:** Breast Cancer Wisconsin (Diagnostic)  
- **Source:** UCI dataset (loaded via `sklearn.datasets.load_breast_cancer`)  
- **Rows:** 569  
- **Features:** 30 numeric features  
- **Target column:** `target`  
  - `0` = malignant  
  - `1` = benign  

---

## 3) Models implemented

1. Logistic Regression *(with StandardScaler via Pipeline)*
2. Decision Tree
3. k-Nearest Neighbors (kNN) *(with StandardScaler via Pipeline)*
4. Naive Bayes (GaussianNB)
5. Random Forest
6. XGBoost

---

## 4) Metrics reported

For each model, the app computes and displays:

- **Accuracy**
- **AUC (ROC-AUC)**
- **Precision**
- **Recall**
- **F1-score**
- **MCC (Matthews Correlation Coefficient)**
- **Confusion Matrix**
- **Classification Report**

---

## 5) Train/Test split

- **Train:** 80%
- **Test:** 20%
- **Stratified split:** Yes
- **Random state:** 42

---

## 6) What gets saved (generated during training)

When you run training (`python app.py --train`) or click **Train & save all models (quick)** in Streamlit, these files are created/updated:

### Models
- `model/logistic_regression.joblib`
- `model/decision_tree.joblib`
- `model/knn.joblib`
- `model/naive_bayes.joblib`
- `model/random_forest.joblib`
- `model/xgboost.joblib`

### Reports
- `reports/metrics.csv` → comparison table for all 6 models
- `reports/test_data_for_upload.csv` → the 20% test split **with** `target` (recommended for upload)
- `reports/train_data.csv` → the 80% train split **with** `target`
- `reports/full_data.csv` → full dataset **with** `target`

---

## 7) How to run

### A) Install dependencies
```bash
pip install -r requirements.txt
```

### B) Train + save models + generate reports
```bash
python app.py --train
```

### C) Run the Streamlit app
```bash
streamlit run app.py
```

---

## 8) Streamlit usage

1. (Recommended) Upload `reports/test_data_for_upload.csv`
2. Select target column as `target`
3. Pick a model from the dropdown
4. Click **Run evaluation**

### Notes
- If you upload a CSV **without** a target column, the app will show:
  - `prediction`
  - `prob_positive`
  (Metrics require a target column.)
- If the app shows **“Saved model not found”**, train once using:
  - `python app.py --train`
  - or the sidebar training button

---

## 9) Results (from `reports/metrics.csv`)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| XGBoost | 0.9561 | 0.9947 | 0.9467 | 0.9861 | 0.9660 | 0.9058 |
| Random Forest | 0.9561 | 0.9934 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |

---

## 10) Model-wise observations (based on the above run)

| Model | Observation |
|---|---|
| Logistic Regression | Best overall AUC and accuracy; scaling helps a lot for stable performance. |
| kNN | Very strong recall (1.0) after scaling; slightly lower precision than LR. |
| Random Forest | Strong and consistent; good AUC with less tuning effort. |
| XGBoost | Competitive AUC; performs well but depends on hyperparameter choices. |
| Naive Bayes | Fast baseline; decent performance despite independence assumptions. |
| Decision Tree | Lowest AUC in this run; tends to overfit without pruning/tuning. |

---

## 11) Project structure

```
ass-2/
│-- app.py
│-- requirements.txt
│-- README.md
│
├─ model/
│   ├─ logistic_regression.joblib
│   ├─ decision_tree.joblib
│   ├─ knn.joblib
│   ├─ naive_bayes.joblib
│   ├─ random_forest.joblib
│   ├─ xgboost.joblib
│
└─ reports/
    ├─ full_data.csv
    ├─ train_data.csv
    ├─ metrics.csv
    ├─ test_data_for_upload.csv
```

---
