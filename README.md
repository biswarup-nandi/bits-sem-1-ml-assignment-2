# ML Assignment 2 — 6 Classifiers Demo (Streamlit)

This project trains and compares **6 classification models** and provides a **Streamlit web app** to upload CSV test data, select a model, and view evaluation outputs (**Accuracy, AUC, Precision, Recall, F1, MCC + Confusion Matrix + Classification Report**).

---

## Dataset

**Default dataset:** Breast Cancer Wisconsin (Diagnostic) dataset (binary classification) from `scikit-learn`.

- **Rows:** 569  
- **Features:** 30 numeric features  
- **Target column name:** `target` (0/1)

The script also creates a ready-to-upload test CSV:

- `reports/test_data_for_upload.csv` (the 20% test split + `target`)

---

## Models Implemented (6)

1. Logistic Regression  
2. Decision Tree  
3. k-Nearest Neighbors (kNN)  
4. Naive Bayes (GaussianNB)  
5. Random Forest  
6. XGBoost  

---

## Train/Test Split

- **Train:** 80%  
- **Test:** 20%  
- **Stratified:** Yes  
- **Random seed:** 42  

---

## How to Run

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Train + save all models + generate reports
```bash
python app.py --train
```

This generates:
- `model/*.joblib` (6 saved trained models)
- `reports/metrics.csv` (comparison table)
- `reports/test_data_for_upload.csv` (upload-ready test dataset)

### 3) Run the Streamlit app
```bash
streamlit run app.py
```

---

## Streamlit App Usage

1. Upload: `reports/test_data_for_upload.csv`
2. Select target column: `target`
3. Choose a model from dropdown
4. Click **Run evaluation**

Outputs shown in the UI:
- Metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion Matrix
- Classification Report

---

## Results (From This Run)

The comparison table below is generated from `reports/metrics.csv`:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| XGBoost | 0.9561 | 0.9947 | 0.9467 | 0.9861 | 0.9660 | 0.9058 |
| Random Forest | 0.9561 | 0.9934 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |

---

## Key Observations

- **Best model by AUC:** **Logistic Regression** (AUC=0.9954, Accuracy=0.9825)
- **Logistic Regression** performed extremely well on this dataset and achieved the highest AUC in this run.
- **XGBoost** and **Random Forest** also achieved strong AUC values, consistent with ensemble-based classifiers.
- **kNN** performed well after standard scaling (used via pipeline).
- **Decision Tree** gave the weakest AUC among the six models in this run, likely due to overfitting/variance.

---

## Project Structure

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
    ├─ metrics.csv
    ├─ test_data_for_upload.csv
```

---

## Notes

- If the Streamlit app shows "Saved model not found", run `python app.py --train` first and start Streamlit from the same project folder.
- Uploaded CSV must contain the same feature columns as the trained dataset; extra columns are ignored and missing columns will raise an error.
