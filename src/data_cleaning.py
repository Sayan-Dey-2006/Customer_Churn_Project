import pandas as pd
import os


# ==========================================
# PROJECT PATH
# ==========================================

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


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(DATA_PATH)


# ==========================================
# BASIC DATASET INFORMATION
# ==========================================

print("=" * 50)
print("CUSTOMER CHURN DATASET")
print("=" * 50)

print("\nDataset loaded successfully!")

print("\nNumber of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])


# ==========================================
# COLUMN NAMES
# ==========================================

print("\nColumn Names:")
print(df.columns.tolist())


# ==========================================
# FIRST 5 ROWS
# ==========================================

print("\nFirst 5 Rows:")
print(df.head())


# ==========================================
# DATA TYPES
# ==========================================

print("\nData Types:")
print(df.dtypes)


# ==========================================
# MISSING VALUES
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================
# DUPLICATE ROWS
# ==========================================

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ==========================================
# CHURN DISTRIBUTION
# ==========================================

if "Churn" in df.columns:

    print("\nChurn Distribution:")
    print(df["Churn"].value_counts())

    print("\nChurn Percentage:")
    print(
        df["Churn"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )


print("\n" + "=" * 50)
print("DATASET CHECK COMPLETED")
print("=" * 50)