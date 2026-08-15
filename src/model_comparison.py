import pandas as pd
import os
import matplotlib.pyplot as plt


# =====================================================
# PATH CONFIGURATION
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

VISUALIZATION_DIR = os.path.join(
    BASE_DIR,
    "visualizations"
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

os.makedirs(
    VISUALIZATION_DIR,
    exist_ok=True
)


# =====================================================
# MODEL PERFORMANCE DATA
# =====================================================

comparison = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Random Forest"
        ],

        "Accuracy": [
            0.8070,
            0.7587
        ],

        "Precision": [
            0.6584,
            0.5318
        ],

        "Recall": [
            0.5668,
            0.7594
        ],

        "F1 Score": [
            0.6092,
            0.6256
        ],

        "ROC-AUC": [
            0.8416,
            0.8413
        ]
    }
)


# =====================================================
# DISPLAY COMPARISON
# =====================================================

print("=" * 70)
print("CUSTOMER CHURN - MODEL COMPARISON")
print("=" * 70)

print(
    "\nModel Performance:"
)

print(
    comparison.to_string(
        index=False
    )
)


# =====================================================
# FIND BEST MODELS
# =====================================================

best_accuracy = comparison.loc[
    comparison["Accuracy"].idxmax()
]

best_precision = comparison.loc[
    comparison["Precision"].idxmax()
]

best_recall = comparison.loc[
    comparison["Recall"].idxmax()
]

best_f1 = comparison.loc[
    comparison["F1 Score"].idxmax()
]

best_roc_auc = comparison.loc[
    comparison["ROC-AUC"].idxmax()
]


print("\n" + "=" * 70)
print("BEST PERFORMANCE")
print("=" * 70)

print(
    f"\nBest Accuracy  : "
    f"{best_accuracy['Model']} "
    f"({best_accuracy['Accuracy']:.4f})"
)

print(
    f"Best Precision : "
    f"{best_precision['Model']} "
    f"({best_precision['Precision']:.4f})"
)

print(
    f"Best Recall    : "
    f"{best_recall['Model']} "
    f"({best_recall['Recall']:.4f})"
)

print(
    f"Best F1 Score  : "
    f"{best_f1['Model']} "
    f"({best_f1['F1 Score']:.4f})"
)

print(
    f"Best ROC-AUC   : "
    f"{best_roc_auc['Model']} "
    f"({best_roc_auc['ROC-AUC']:.4f})"
)


# =====================================================
# SAVE COMPARISON REPORT
# =====================================================

report_path = os.path.join(
    REPORT_DIR,
    "model_comparison.csv"
)

comparison.to_csv(
    report_path,
    index=False
)

print(
    "\nComparison report saved:"
)

print(
    report_path
)


# =====================================================
# CREATE COMPARISON CHART
# =====================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

x = range(
    len(metrics)
)

width = 0.35

plt.figure(
    figsize=(11, 6)
)

plt.bar(
    [i - width / 2 for i in x],
    comparison.iloc[0][metrics],
    width=width,
    label="Logistic Regression"
)

plt.bar(
    [i + width / 2 for i in x],
    comparison.iloc[1][metrics],
    width=width,
    label="Random Forest"
)

plt.xticks(
    list(x),
    metrics
)

plt.ylim(
    0,
    1
)

plt.ylabel(
    "Score"
)

plt.xlabel(
    "Evaluation Metric"
)

plt.title(
    "Customer Churn Model Comparison"
)

plt.legend()

plt.tight_layout()


chart_path = os.path.join(
    VISUALIZATION_DIR,
    "model_comparison.png"
)

plt.savefig(
    chart_path
)

plt.show()


# =====================================================
# FINAL RECOMMENDATION
# =====================================================

print("\n" + "=" * 70)
print("MODEL RECOMMENDATION")
print("=" * 70)

print(
    "\nFor general balanced performance:"
)

print(
    "Logistic Regression is stronger in "
    "Accuracy, Precision and ROC-AUC."
)

print(
    "\nFor identifying more potential churn customers:"
)

print(
    "Random Forest is stronger because "
    "it has significantly higher Recall."
)

print(
    "\nRandom Forest Recall:",
    f"{best_recall['Recall']:.4f}"
)

print(
    "\nLogistic Regression Accuracy:",
    f"{best_accuracy['Accuracy']:.4f}"
)


# =====================================================
# FINAL
# =====================================================

print("\n" + "=" * 70)

print(
    "MODEL COMPARISON COMPLETED"
)

print("=" * 70)