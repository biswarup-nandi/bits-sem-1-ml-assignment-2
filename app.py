# -----------------------------------------------------------------------------
# ML Assignment 2
# -----------------------------------------------------------------------------
# NAME: BISWARUP NANDI
# STUDENT ID: 2025AA05115
# MAIL ID: 2025aa05115@wilp.bits-pilani.ac.in
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 1) Train + save models:
#       python app.py --train
#
# 2) Run Streamlit app:
#       streamlit run app.py
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import time
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

import joblib

try:
    from xgboost import XGBClassifier
except Exception as e:
    XGBClassifier = None
    _XGB_IMPORT_ERROR = e

RANDOM_STATE = 42
MODEL_DIR = Path("model")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
}

# -------------------- Small utility for progress printing --------------------
def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def lap(self) -> float:
        return time.perf_counter() - self.t0

# -------------------- Dataset helpers --------------------
def load_default_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    y = data.target.copy()
    return X, y

def validate_dataset(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    *,
    min_rows: int = 500,
    min_features: int = 12
) -> None:
    if X.shape[1] < min_features:
        raise ValueError(f"Dataset must have at least {min_features} features. Found: {X.shape[1]}")
    if X.shape[0] < min_rows:
        raise ValueError(f"Dataset must have at least {min_rows} instances. Found: {X.shape[0]}")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

# -------------------- Models --------------------
def build_models() -> Dict[str, object]:
    if XGBClassifier is None:
        raise ImportError(
            f"XGBoost is required but could not be imported.\nError: {_XGB_IMPORT_ERROR}\n"
            f"Fix: pip install xgboost"
        )

    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7))
        ]),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        ),
    }

# -------------------- Metrics --------------------
def positive_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        return scores
    raise ValueError("Model has neither predict_proba nor decision_function.")

def compute_metrics(model, X_te: pd.DataFrame, y_te: pd.Series) -> dict:
    y_pred = model.predict(X_te)
    y_prob = positive_proba(model, X_te)

    return {
        "Accuracy": float(accuracy_score(y_te, y_pred)),
        "AUC": float(roc_auc_score(y_te, y_prob)),
        "Precision": float(precision_score(y_te, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_te, y_pred, zero_division=0)),
        "F1": float(f1_score(y_te, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_te, y_pred)),
        "ConfusionMatrix": confusion_matrix(y_te, y_pred),
        "ClassificationReport": classification_report(y_te, y_pred, zero_division=0),
    }

# -------------------- Train + Save --------------------
def train_and_save_all(save_test_csv: bool = True) -> pd.DataFrame:
    overall = Timer()
    log("STEP 1/6: Loading default dataset (Breast Cancer)...")
    X, y = load_default_dataset()
    log(f"Loaded dataset: X={X.shape}, y={y.shape} (rows={len(X)}, features={X.shape[1]})")

    log("STEP 2/6: Validating dataset constraints (>=500 rows, >=12 features)...")
    validate_dataset(X, y, min_rows=500, min_features=12)  # training must satisfy assignment
    log("Validation OK.")

    log("STEP 3/6: Train-test split (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    log(f"Split done: Train={X_train.shape}, Test={X_test.shape}")

    if save_test_csv:
        log("Saving test CSV for Streamlit upload...")
        test_df = X_test.copy()
        test_df["target"] = y_test.values
        test_path = REPORT_DIR / "test_data_for_upload.csv"
        test_df.to_csv(test_path, index=False)
        log(f"Saved test CSV: {test_path.resolve()}")

    log("STEP 4/6: Building 6 models (LR, DT, kNN, NB, RF, XGBoost)...")
    models = build_models()
    log(f"Models ready: {list(models.keys())}")

    log("STEP 5/6: Training + evaluating each model...")
    rows = []
    for i, (name, model) in enumerate(models.items(), start=1):
        t = Timer()
        log(f"  [{i}/6] START: {name}")
        log(f"  [{i}/6] Fitting model...")
        model.fit(X_train, y_train)
        log(f"  [{i}/6] Fit done. Computing metrics...")
        m = compute_metrics(model, X_test, y_test)

        log(
            f"  [{i}/6] Metrics: "
            f"Acc={m['Accuracy']:.4f}, AUC={m['AUC']:.4f}, "
            f"P={m['Precision']:.4f}, R={m['Recall']:.4f}, F1={m['F1']:.4f}, MCC={m['MCC']:.4f}"
        )

        out_path = MODEL_DIR / MODEL_FILES[name]
        log(f"  [{i}/6] Saving model -> {out_path} ...")
        joblib.dump(model, out_path)
        log(f"  [{i}/6] Saved. Time taken: {t.lap():.2f}s")

        rows.append({
            "ML Model Name": name,
            "Accuracy": m["Accuracy"],
            "AUC": m["AUC"],
            "Precision": m["Precision"],
            "Recall": m["Recall"],
            "F1": m["F1"],
            "MCC": m["MCC"],
        })

    log("STEP 6/6: Writing comparison table (reports/metrics.csv)...")
    metrics_df = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    metrics_csv = REPORT_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    log(f"Saved metrics CSV: {metrics_csv.resolve()}")

    log(f"ALL DONE! Total time: {overall.lap():.2f}s")
    return metrics_df

def load_saved_model(model_name: str):
    path = MODEL_DIR / MODEL_FILES[model_name]
    if not path.exists():
        return None
    return joblib.load(path)

# -------------------- Streamlit App --------------------
def run_streamlit_app() -> None:
    import streamlit as st
    import matplotlib.pyplot as plt

    st.set_page_config(page_title="ML Assignment 2 - Classifiers", layout="wide")
    st.title("Machine Learning Assignment 2 — 6 Classifiers Demo for Breast Cancer Dataset")

    X_default, y_default = load_default_dataset()
    default_cols = list(X_default.columns)

    with st.sidebar:
        st.header("Controls")
        uploaded = st.file_uploader("Upload CSV (test data recommended)", type=["csv"])
        model_name = st.selectbox("Model", list(MODEL_FILES.keys()))
        train_now = st.button("Train & save all models (quick)")
        run_eval = st.button("Run evaluation")

    if train_now:
        try:
            metrics_df = train_and_save_all(save_test_csv=True)
            st.success("Training complete. Models saved in ./model and metrics saved in ./reports.")
            st.dataframe(metrics_df, use_container_width=True)
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.subheader("Uploaded CSV preview")
        st.dataframe(df.head(10), use_container_width=True)

        candidates = [c for c in df.columns if c.lower() in ("target", "label", "y")]
        target_col = st.selectbox(
            "Target column (required for metrics)",
            options=(["<None>"] + candidates + [c for c in df.columns if c not in candidates]),
            index=0
        )
        if target_col == "<None>":
            target_col = None

        if target_col is not None:
            y = df[target_col].astype(int)
            X = df.drop(columns=[target_col])
        else:
            y = None
            X = df

        missing = [c for c in default_cols if c not in X.columns]
        if missing:
            st.error(f"Missing required feature columns: {missing}")
            st.stop()

        extra = [c for c in X.columns if c not in default_cols]
        if extra:
            st.info(f"Extra columns ignored: {extra}")
            X = X[default_cols]

        validate_dataset(X, y if y is not None else None, min_rows=1, min_features=12)

    else:
        st.subheader("Default dataset in use (Breast Cancer)")
        preview = X_default.head(8).copy()
        preview["target"] = y_default.head(8).values
        st.dataframe(preview, use_container_width=True)
        X, y = X_default, y_default

        validate_dataset(X, y, min_rows=1, min_features=12)

    model = load_saved_model(model_name)
    if model is None:
        st.warning("Saved model not found. Train once using python app.py --train or sidebar button.")
        st.stop()

    if run_eval:
        st.subheader(f"Results — {model_name}")

        if y is None:
            st.warning("No target provided. Showing predictions only (metrics need target).")
            preds = model.predict(X)
            prob = positive_proba(model, X)
            out = X.copy()
            out["prediction"] = preds
            out["prob_positive"] = prob
            st.dataframe(out.head(50), use_container_width=True)
            st.stop()

        m = compute_metrics(model, X, y)

        c1, c2 = st.columns(2)
        with c1:
            st.write("### Evaluation metrics")
            st.dataframe(pd.DataFrame([{
                "Accuracy": m["Accuracy"],
                "AUC": m["AUC"],
                "Precision": m["Precision"],
                "Recall": m["Recall"],
                "F1": m["F1"],
                "MCC": m["MCC"],
            }]), use_container_width=True)

        with c2:
            st.write("### Confusion Matrix")
            cm = m["ConfusionMatrix"]
            fig, ax = plt.subplots()
            ax.imshow(cm)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center")
            st.pyplot(fig)

        st.write("### Classification Report")
        st.code(m["ClassificationReport"])

        st.caption("Tip: Use reports/test_data_for_upload.csv for upload testing.")

# -------------------- Main --------------------
def running_in_streamlit() -> bool:
    """
    FIX #1: Detect Streamlit execution so `streamlit run app.py` launches the UI.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train + save all 6 models and metrics.")
    args, _ = parser.parse_known_args()  # safe if any extra args appear

    if args.train:
        df = train_and_save_all(save_test_csv=True)
        print("\nTraining complete. Comparison table (sorted by AUC):\n")
        print(df.to_string(index=False))
        print("\nSaved:")
        print(f"- Models: {MODEL_DIR.resolve()}")
        print(f"- Metrics CSV: {(REPORT_DIR / 'metrics.csv').resolve()}")
        print(f"- Test CSV for upload: {(REPORT_DIR / 'test_data_for_upload.csv').resolve()}")
        return

    # If Streamlit is running this script, launch the UI
    if running_in_streamlit():
        run_streamlit_app()
        return

    # If user runs `python app.py` without --train, show instructions
    print("Run one of these:")
    print("1) python app.py --train")
    print("2) streamlit run app.py")

if __name__ == "__main__":
    main()
