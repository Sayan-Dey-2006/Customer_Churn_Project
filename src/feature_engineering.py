import pandas as pd
import os


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
    "customer_churn.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed_churn.csv"
)


# =====================================================
# LOAD DATASET
# =====================================================

print("=" * 60)
print("CUSTOMER CHURN - FEATURE ENGINEERING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Original shape:", df.shape)


# =====================================================
# REMOVE CUSTOMER ID
# =====================================================

if "customerID" in df.columns:

    df = df.drop(
        columns=["customerID"]
    )

    print("\nRemoved customerID column.")


# =====================================================
# CONVERT TOTAL CHARGES
# =====================================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nConverted TotalCharges to numeric.")


# =====================================================
# HANDLE MISSING VALUES
# =====================================================

missing_before = df.isnull().sum().sum()

print(
    "\nMissing values before cleaning:",
    missing_before
)

# Fill TotalCharges missing values
# using median

if df["TotalCharges"].isnull().sum() > 0:

    median_total_charges = df[
        "TotalCharges"
    ].median()

    df["TotalCharges"] = df[
        "TotalCharges"
    ].fillna(
        median_total_charges
    )


missing_after = df.isnull().sum().sum()

print(
    "Missing values after cleaning:",
    missing_after
)


# =====================================================
# CONVERT CHURN TO 0 / 1
# =====================================================

df["Churn"] = df["Churn"].map(
    {
        "No": 0,
        "Yes": 1
    }
)

print("\nConverted Churn:")
print("No  = 0")
print("Yes = 1")


# =====================================================
# ONE-HOT ENCODING
# =====================================================

categorical_columns = df.select_dtypes(
    include=["object"]
).columns.tolist()

print(
    "\nCategorical columns:"
)

print(
    categorical_columns
)

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)


# =====================================================
# FINAL DATASET INFORMATION
# =====================================================

print(
    "\nFinal dataset shape:",
    df.shape
)

print(
    "\nFinal columns:"
)

for column in df.columns:

    print(
        "-",
        column
    )


# =====================================================
# CHECK MISSING VALUES
# =====================================================

print(
    "\nFinal missing values:"
)

print(
    df.isnull().sum().sum()
)


# =====================================================
# SAVE PROCESSED DATASET
# =====================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    "\nProcessed dataset saved successfully!"
)

print(
    "Location:",
    OUTPUT_PATH
)


# =====================================================
# FINAL MESSAGE
# =====================================================

print("\n" + "=" * 60)

print(
    "FEATURE ENGINEERING COMPLETED"
)

print("=" * 60)