import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


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
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    VISUALIZATION_DIR,
    exist_ok=True
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "churn_model.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)


# =====================================================
# LOAD PROCESSED DATA
# =====================================================

print("=" * 60)
print("CUSTOMER CHURN - MACHINE LEARNING")
print("=" * 60)

print("\nLoading processed dataset...")

df = pd.read_csv(DATA_PATH)

print(
    "Dataset loaded successfully!"
)

print(
    "Dataset shape:",
    df.shape
)


# =====================================================
# SEPARATE FEATURES AND TARGET
# =====================================================

X = df.drop(
    columns=["Churn"]
)

y = df["Churn"]


print(
    "\nFeatures:",
    X.shape[1]
)

print(
    "Target:",
    "Churn"
)


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


print(
    "\nTraining samples:",
    X_train.shape[0]
)

print(
    "Testing samples:",
    X_test.shape[0]
)


# =====================================================
# FEATURE SCALING
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# =====================================================
# LOGISTIC REGRESSION MODEL
# =====================================================

print(
    "\nTraining Logistic Regression model..."
)

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)


print(
    "Model training completed!"
)


# =====================================================
# PREDICTIONS
# =====================================================

y_pred = model.predict(
    X_test_scaled
)

y_probability = model.predict_proba(
    X_test_scaled
)[:, 1]


# =====================================================
# MODEL EVALUATION
# =====================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    f"\nAccuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Not Churn",
            "Churn"
        ]
    )
)


# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print(
    "\nConfusion Matrix:"
)

print(cm)


# =====================================================
# CONFUSION MATRIX VISUALIZATION
# =====================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Customer Churn - Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["Not Churn", "Churn"]
)

plt.yticks(
    [0, 1],
    ["Not Churn", "Churn"]
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

confusion_matrix_path = os.path.join(
    VISUALIZATION_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path
)

plt.show()


# =====================================================
# FEATURE IMPORTANCE
# =====================================================

coefficients = model.coef_[0]

feature_importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Coefficient": coefficients,
        "Absolute_Impact": np.abs(
            coefficients
        )
    }
)

feature_importance = feature_importance.sort_values(
    by="Absolute_Impact",
    ascending=False
)


print(
    "\nTop 15 Important Features:"
)

print(
    feature_importance.head(15).to_string(
        index=False
    )
)


# =====================================================
# SAVE FEATURE IMPORTANCE
# =====================================================

feature_importance_path = os.path.join(
    BASE_DIR,
    "reports",
    "feature_importance.csv"
)

os.makedirs(
    os.path.dirname(
        feature_importance_path
    ),
    exist_ok=True
)

feature_importance.to_csv(
    feature_importance_path,
    index=False
)


# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)


print(
    "\nModel saved successfully!"
)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Scaler:",
    SCALER_PATH
)


# =====================================================
# FINAL MESSAGE
# =====================================================

print("\n" + "=" * 60)

print(
    "MACHINE LEARNING COMPLETED"
)

print("=" * 60)