import pandas as pd
import matplotlib.pyplot as plt
import os


# =====================================================
# PATH
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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "visualizations"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(DATA_PATH)


# =====================================================
# DATA CLEANING FOR ANALYSIS
# =====================================================

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove rows where TotalCharges is missing
df = df.dropna(subset=["TotalCharges"])


# =====================================================
# BASIC INFORMATION
# =====================================================

print("=" * 60)
print("CUSTOMER CHURN - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nTotal Customers:", len(df))

print(
    "Churned Customers:",
    (df["Churn"] == "Yes").sum()
)

print(
    "Customers Stayed:",
    (df["Churn"] == "No").sum()
)

print("\nChurn Rate:")

churn_rate = (
    df["Churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(churn_rate)


# =====================================================
# 1. CHURN DISTRIBUTION
# =====================================================

plt.figure(figsize=(7, 5))

df["Churn"].value_counts().plot(
    kind="bar"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "churn_distribution.png"
    )
)

plt.show()


# =====================================================
# 2. CHURN BY GENDER
# =====================================================

gender_churn = pd.crosstab(
    df["gender"],
    df["Churn"]
)

print("\nChurn by Gender:")
print(gender_churn)

gender_churn.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Churn by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "churn_by_gender.png"
    )
)

plt.show()


# =====================================================
# 3. CHURN BY CONTRACT
# =====================================================

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"]
)

print("\nChurn by Contract:")
print(contract_churn)

contract_churn.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Churn by Contract Type")
plt.xlabel("Contract")
plt.ylabel("Number of Customers")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "churn_by_contract.png"
    )
)

plt.show()


# =====================================================
# 4. CHURN BY INTERNET SERVICE
# =====================================================

internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"]
)

print("\nChurn by Internet Service:")
print(internet_churn)

internet_churn.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "churn_by_internet_service.png"
    )
)

plt.show()


# =====================================================
# 5. CHURN BY PAYMENT METHOD
# =====================================================

payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"]
)

print("\nChurn by Payment Method:")
print(payment_churn)

payment_churn.plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")

plt.xticks(rotation=25)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "churn_by_payment_method.png"
    )
)

plt.show()


# =====================================================
# 6. CHURN BY SENIOR CITIZEN
# =====================================================

senior_churn = pd.crosstab(
    df["SeniorCitizen"],
    df["Churn"]
)

print("\nChurn by Senior Citizen:")
print(senior_churn)

senior_churn.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Churn by Senior Citizen")
plt.xlabel("Senior Citizen (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "churn_by_senior_citizen.png"
    )
)

plt.show()


# =====================================================
# 7. CHURN BY PARTNER
# =====================================================

partner_churn = pd.crosstab(
    df["Partner"],
    df["Churn"]
)

print("\nChurn by Partner:")
print(partner_churn)

partner_churn.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Churn by Partner Status")
plt.xlabel("Partner")
plt.ylabel("Number of Customers")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "churn_by_partner.png"
    )
)

plt.show()


# =====================================================
# 8. CHURN BY DEPENDENTS
# =====================================================

dependents_churn = pd.crosstab(
    df["Dependents"],
    df["Churn"]
)

print("\nChurn by Dependents:")
print(dependents_churn)

dependents_churn.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Churn by Dependents")
plt.xlabel("Dependents")
plt.ylabel("Number of Customers")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "churn_by_dependents.png"
    )
)

plt.show()


# =====================================================
# 9. MONTHLY CHARGES VS CHURN
# =====================================================

plt.figure(figsize=(8, 5))

df.boxplot(
    column="MonthlyCharges",
    by="Churn"
)

plt.title("Monthly Charges vs Churn")
plt.suptitle("")

plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "monthly_charges_vs_churn.png"
    )
)

plt.show()


# =====================================================
# 10. TENURE VS CHURN
# =====================================================

plt.figure(figsize=(8, 5))

df.boxplot(
    column="tenure",
    by="Churn"
)

plt.title("Tenure vs Churn")
plt.suptitle("")

plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "tenure_vs_churn.png"
    )
)

plt.show()


# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n" + "=" * 60)
print("EDA COMPLETED")
print("=" * 60)

print("\nCharts saved inside:")
print(OUTPUT_DIR)

print("\nGenerated charts:")

for file in os.listdir(OUTPUT_DIR):

    if file.endswith(".png"):

        print("-", file)

print("\nNext Step:")
print("Feature Engineering and Machine Learning")