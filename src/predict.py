import os
import joblib
import pandas as pd


# =====================================================
# PATH CONFIGURATION
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed_churn.csv"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "random_forest_model.pkl"
)


# =====================================================
# LOAD MODEL
# =====================================================

print("=" * 65)
print("CUSTOMER CHURN PREDICTION SYSTEM")
print("=" * 65)

print("\nLoading model...")

model = joblib.load(
    MODEL_PATH
)

print("Model loaded successfully!")


# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(
    DATA_PATH
)

feature_columns = [
    column
    for column in df.columns
    if column != "Churn"
]


# =====================================================
# INPUT FUNCTION
# =====================================================

def get_yes_no(question):

    while True:

        answer = input(
            question + " (Yes/No): "
        ).strip().lower()

        if answer in ["yes", "y"]:
            return 1

        if answer in ["no", "n"]:
            return 0

        print(
            "Please enter Yes or No."
        )


def get_number(
    question,
    minimum=None,
    maximum=None,
    integer=False
):

    while True:

        try:

            value = float(
                input(question).strip()
            )

            if minimum is not None and value < minimum:
                print(
                    f"Value must be at least {minimum}."
                )
                continue

            if maximum is not None and value > maximum:
                print(
                    f"Value must be at most {maximum}."
                )
                continue

            if integer:
                return int(value)

            return value

        except ValueError:

            print(
                "Please enter a valid number."
            )


# =====================================================
# GET CUSTOMER INFORMATION
# =====================================================

print("\nEnter customer information")
print("-" * 65)

tenure = get_number(
    "Tenure (months): ",
    minimum=0,
    maximum=100,
    integer=True
)

monthly_charges = get_number(
    "Monthly Charges: ",
    minimum=0
)

total_charges = get_number(
    "Total Charges: ",
    minimum=0
)

gender_male = get_yes_no(
    "Gender Male?"
)

partner_yes = get_yes_no(
    "Has Partner?"
)

dependents_yes = get_yes_no(
    "Has Dependents?"
)

phone_service_yes = get_yes_no(
    "Has Phone Service?"
)

multiple_lines = input(
    "Multiple Lines "
    "(Yes/No/No phone service): "
).strip().lower()

internet_service = input(
    "Internet Service "
    "(DSL/Fiber optic/No): "
).strip().lower()

online_security = input(
    "Online Security "
    "(Yes/No/No internet service): "
).strip().lower()

online_backup = input(
    "Online Backup "
    "(Yes/No/No internet service): "
).strip().lower()

device_protection = input(
    "Device Protection "
    "(Yes/No/No internet service): "
).strip().lower()

tech_support = input(
    "Tech Support "
    "(Yes/No/No internet service): "
).strip().lower()

streaming_tv = input(
    "Streaming TV "
    "(Yes/No/No internet service): "
).strip().lower()

streaming_movies = input(
    "Streaming Movies "
    "(Yes/No/No internet service): "
).strip().lower()

contract = input(
    "Contract "
    "(Month-to-month/One year/Two year): "
).strip().lower()

paperless_billing_yes = get_yes_no(
    "Paperless Billing?"
)

payment_method = input(
    "Payment Method "
    "(Bank transfer (automatic)/"
    "Credit card (automatic)/"
    "Electronic check/"
    "Mailed check): "
).strip().lower()


# =====================================================
# CREATE CUSTOMER DATA
# =====================================================

customer = {
    "SeniorCitizen": 0,

    "tenure": tenure,

    "MonthlyCharges": monthly_charges,

    "TotalCharges": total_charges,

    "gender_Male": gender_male,

    "Partner_Yes": partner_yes,

    "Dependents_Yes": dependents_yes,

    "PhoneService_Yes": phone_service_yes,

    "MultipleLines_No phone service":
        1 if multiple_lines == "no phone service" else 0,

    "MultipleLines_Yes":
        1 if multiple_lines == "yes" else 0,

    "InternetService_Fiber optic":
        1 if internet_service == "fiber optic" else 0,

    "InternetService_No":
        1 if internet_service == "no" else 0,

    "OnlineSecurity_No internet service":
        1 if online_security == "no internet service" else 0,

    "OnlineSecurity_Yes":
        1 if online_security == "yes" else 0,

    "OnlineBackup_No internet service":
        1 if online_backup == "no internet service" else 0,

    "OnlineBackup_Yes":
        1 if online_backup == "yes" else 0,

    "DeviceProtection_No internet service":
        1 if device_protection == "no internet service" else 0,

    "DeviceProtection_Yes":
        1 if device_protection == "yes" else 0,

    "TechSupport_No internet service":
        1 if tech_support == "no internet service" else 0,

    "TechSupport_Yes":
        1 if tech_support == "yes" else 0,

    "StreamingTV_No internet service":
        1 if streaming_tv == "no internet service" else 0,

    "StreamingTV_Yes":
        1 if streaming_tv == "yes" else 0,

    "StreamingMovies_No internet service":
        1 if streaming_movies == "no internet service" else 0,

    "StreamingMovies_Yes":
        1 if streaming_movies == "yes" else 0,

    "Contract_One year":
        1 if contract == "one year" else 0,

    "Contract_Two year":
        1 if contract == "two year" else 0,

    "PaperlessBilling_Yes":
        paperless_billing_yes,

    "PaymentMethod_Credit card (automatic)":
        1 if payment_method == "credit card (automatic)" else 0,

    "PaymentMethod_Electronic check":
        1 if payment_method == "electronic check" else 0,

    "PaymentMethod_Mailed check":
        1 if payment_method == "mailed check" else 0
}


# =====================================================
# CREATE DATAFRAME
# =====================================================

customer_df = pd.DataFrame(
    [customer]
)

customer_df = customer_df.reindex(
    columns=feature_columns,
    fill_value=0
)


# =====================================================
# PREDICTION
# =====================================================

prediction = model.predict(
    customer_df
)[0]

probability = model.predict_proba(
    customer_df
)[0][1]

churn_percentage = probability * 100


# =====================================================
# RISK LEVEL
# =====================================================

if churn_percentage >= 70:

    risk = "HIGH"

elif churn_percentage >= 40:

    risk = "MEDIUM"

else:

    risk = "LOW"


# =====================================================
# RESULT
# =====================================================

print("\n" + "=" * 65)
print("CUSTOMER CHURN PREDICTION")
print("=" * 65)

print(
    f"\nChurn Probability : "
    f"{churn_percentage:.2f}%"
)

print(
    f"Risk Level        : {risk}"
)

if prediction == 1:

    print(
        "Prediction        : Customer is likely to churn."
    )

else:

    print(
        "Prediction        : Customer is likely to stay."
    )


# =====================================================
# BUSINESS RECOMMENDATION
# =====================================================

print("\n" + "-" * 65)
print("BUSINESS RECOMMENDATION")
print("-" * 65)

if risk == "HIGH":

    print(
        "Immediate retention action is recommended."
    )

    print(
        "Consider offering personalized support, "
        "contract incentives, or suitable service benefits."
    )

elif risk == "MEDIUM":

    print(
        "Customer should be monitored closely."
    )

    print(
        "Consider targeted engagement and "
        "customer satisfaction campaigns."
    )

else:

    print(
        "Customer currently has relatively low churn risk."
    )

    print(
        "Continue normal customer engagement."
    )


print("\n" + "=" * 65)
print("PREDICTION COMPLETED")
print("=" * 65)