import pandas as pd
import numpy as np
import os
import joblib

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)


# =====================================================
# PATH CONFIGURATION
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed_churn.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

VISUALIZATION_DIR = os.path.join(
    BASE_DIR,
    "visualizations"
)

os.makedirs(
    VISUALIZATION_DIR,
    exist_ok=True
)


LOGISTIC_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "churn_model.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

RANDOM_FOREST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "random_forest_model.pkl"
)


# =====================================================
# HEADER
# =====================================================

print("=" * 65)
print("CUSTOMER CHURN - MODEL EVALUATION")
print("=" * 65)


# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading processed dataset...")

df = pd.read_csv(
    DATA_PATH
)

print(
    "Dataset loaded successfully!"
)

print(
    "Dataset shape:",
    df.shape
)


# =====================================================
# FEATURES AND TARGET
# =====================================================

X = df.drop(
    columns=["Churn"]
)

y = df["Churn"]


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =====================================================
# LOAD MODELS
# =====================================================

print("\nLoading trained models...")

logistic_model = joblib.load(
    LOGISTIC_MODEL_PATH
)

random_forest_model = joblib.load(
    RANDOM_FOREST_MODEL_PATH
)

scaler = joblib.load(
    SCALER_PATH
)

print(
    "Models loaded successfully!"
)


# =====================================================
# SCALE DATA FOR LOGISTIC REGRESSION
# =====================================================

X_test_scaled = scaler.transform(
    X_test
)


# =====================================================
# PREDICTIONS
# =====================================================

logistic_prediction = logistic_model.predict(
    X_test_scaled
)

logistic_probability = logistic_model.predict_proba(
    X_test_scaled
)[:, 1]


random_forest_prediction = random_forest_model.predict(
    X_test
)

random_forest_probability = random_forest_model.predict_proba(
    X_test
)[:, 1]


# =====================================================
# CONFUSION MATRIX - LOGISTIC REGRESSION
# =====================================================

print("\nCreating Logistic Regression confusion matrix...")

cm_logistic = confusion_matrix(
    y_test,
    logistic_prediction
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_logistic,
    display_labels=[
        "Not Churn",
        "Churn"
    ]
)

disp.plot()

plt.title(
    "Logistic Regression - Confusion Matrix"
)

plt.tight_layout()

logistic_cm_path = os.path.join(
    VISUALIZATION_DIR,
    "logistic_confusion_matrix.png"
)

plt.savefig(
    logistic_cm_path
)

plt.close()


# =====================================================
# CONFUSION MATRIX - RANDOM FOREST
# =====================================================

print(
    "Creating Random Forest confusion matrix..."
)

cm_rf = confusion_matrix(
    y_test,
    random_forest_prediction
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=[
        "Not Churn",
        "Churn"
    ]
)

disp.plot()

plt.title(
    "Random Forest - Confusion Matrix"
)

plt.tight_layout()

rf_cm_path = os.path.join(
    VISUALIZATION_DIR,
    "random_forest_confusion_matrix.png"
)

plt.savefig(
    rf_cm_path
)

plt.close()


# =====================================================
# ROC CURVE
# =====================================================

print(
    "Creating ROC curve..."
)

fpr_logistic, tpr_logistic, _ = roc_curve(
    y_test,
    logistic_probability
)

fpr_rf, tpr_rf, _ = roc_curve(
    y_test,
    random_forest_probability
)


auc_logistic = auc(
    fpr_logistic,
    tpr_logistic
)

auc_rf = auc(
    fpr_rf,
    tpr_rf
)


plt.figure(
    figsize=(9, 7)
)

plt.plot(
    fpr_logistic,
    tpr_logistic,
    label=f"Logistic Regression (AUC = {auc_logistic:.4f})"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC = {auc_rf:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Customer Churn - ROC Curve"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


roc_path = os.path.join(
    VISUALIZATION_DIR,
    "roc_curve.png"
)

plt.savefig(
    roc_path
)

plt.close()


# =====================================================
# PRINT RESULTS
# =====================================================

print("\n" + "=" * 65)
print("MODEL EVALUATION RESULTS")
print("=" * 65)

print(
    "\nLogistic Regression Confusion Matrix:"
)

print(
    cm_logistic
)

print(
    "\nRandom Forest Confusion Matrix:"
)

print(
    cm_rf
)

print(
    f"\nLogistic Regression ROC-AUC: "
    f"{auc_logistic:.4f}"
)

print(
    f"Random Forest ROC-AUC: "
    f"{auc_rf:.4f}"
)


# =====================================================
# SAVE EVALUATION REPORT
# =====================================================

evaluation_report = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Random Forest"
        ],

        "ROC-AUC": [
            auc_logistic,
            auc_rf
        ]
    }
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

evaluation_path = os.path.join(
    REPORT_DIR,
    "model_evaluation.csv"
)

evaluation_report.to_csv(
    evaluation_path,
    index=False
)


# =====================================================
# FINAL
# =====================================================

print("\n" + "=" * 65)

print(
    "MODEL EVALUATION COMPLETED"
)

print("=" * 65)

print(
    "\nGenerated files:"
)

print(
    "- logistic_confusion_matrix.png"
)

print(
    "- random_forest_confusion_matrix.png"
)

print(
    "- roc_curve.png"
)

print(
    "- model_evaluation.csv"
)

print(
    "\nAll files saved successfully!"
)