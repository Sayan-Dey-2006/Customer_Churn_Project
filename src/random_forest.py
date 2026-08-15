import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
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

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "random_forest_model.pkl"
)


# =====================================================
# LOAD DATA
# =====================================================

print("=" * 60)
print("CUSTOMER CHURN - RANDOM FOREST")
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

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# =====================================================
# RANDOM FOREST MODEL
# =====================================================

print(
    "\nTraining Random Forest model..."
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print(
    "Model training completed!"
)


# =====================================================
# PREDICTIONS
# =====================================================

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# =====================================================
# EVALUATION
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
print("RANDOM FOREST PERFORMANCE")
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
# FEATURE IMPORTANCE
# =====================================================

feature_importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }
)

feature_importance = feature_importance.sort_values(
    by="Importance",
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

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

REPORT_PATH = os.path.join(
    REPORT_DIR,
    "random_forest_feature_importance.csv"
)

feature_importance.to_csv(
    REPORT_PATH,
    index=False
)


# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(
    model,
    MODEL_PATH
)

print(
    "\nRandom Forest model saved successfully!"
)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Feature importance:",
    REPORT_PATH
)


# =====================================================
# FINAL
# =====================================================

print("\n" + "=" * 60)

print(
    "RANDOM FOREST TRAINING COMPLETED"
)

print("=" * 60)