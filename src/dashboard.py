import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =========================================================
# PATH CONFIGURATION
# =========================================================

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

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest_model.pkl"
)

COMPARISON_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "model_comparison.csv"
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        DATA_PATH
    )


@st.cache_resource
def load_model():

    return joblib.load(
        MODEL_PATH
    )


df = load_data()
model = load_model()


# =========================================================
# TITLE
# =========================================================

st.title(
    "📊 Customer Churn Analytics Dashboard"
)

st.write(
    "Analyze customer behavior and predict "
    "customer churn using Machine Learning."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "📌 Navigation"
)

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Churn Analysis",
        "Model Performance",
        "Churn Prediction"
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.header(
        "📈 Business Overview"
    )

    total_customers = len(df)

    churned = int(
        (df["Churn"] == 1).sum()
    )

    stayed = int(
        (df["Churn"] == 0).sum()
    )

    churn_rate = (
        churned / total_customers
    ) * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

    with col2:

        st.metric(
            "Churned Customers",
            f"{churned:,}"
        )

    with col3:

        st.metric(
            "Customers Stayed",
            f"{stayed:,}"
        )

    with col4:

        st.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%"
        )

    st.divider()

    # -----------------------------------------------------
    # CHURN DISTRIBUTION
    # -----------------------------------------------------

    st.subheader(
        "Customer Churn Distribution"
    )

    churn_counts = df["Churn"].value_counts()

    labels = [
        "Stayed",
        "Churned"
    ]

    values = [
        churn_counts.get(0, 0),
        churn_counts.get(1, 0)
    ]

    fig, ax = plt.subplots()

    ax.bar(
        labels,
        values
    )

    ax.set_ylabel(
        "Number of Customers"
    )

    ax.set_title(
        "Churn vs Non-Churn Customers"
    )

    st.pyplot(
        fig
    )

    plt.close(fig)


# =========================================================
# CHURN ANALYSIS
# =========================================================

elif page == "Churn Analysis":

    st.header(
        "🔍 Customer Churn Analysis"
    )

    # -----------------------------------------------------
    # CONTRACT
    # -----------------------------------------------------

    st.subheader(
        "Churn by Contract"
    )

    contract_data = pd.crosstab(
        df["Contract_One year"],
        df["Churn"]
    )

    contract_labels = [
        "Not One-Year",
        "One-Year"
    ]

    contract_no = [
        contract_data.get(0, pd.Series()).get(0, 0),
        contract_data.get(1, pd.Series()).get(0, 0)
    ]

    contract_yes = [
        contract_data.get(0, pd.Series()).get(1, 0),
        contract_data.get(1, pd.Series()).get(1, 0)
    ]

    contract_df = pd.DataFrame(
        {
            "Stayed": contract_no,
            "Churned": contract_yes
        },
        index=contract_labels
    )

    st.bar_chart(
        contract_df
    )

    st.divider()

    # -----------------------------------------------------
    # INTERNET SERVICE
    # -----------------------------------------------------

    st.subheader(
        "Internet Service and Churn"
    )

    internet_df = pd.DataFrame(
        {
            "Fiber Optic": [
                (
                    (df["InternetService_Fiber optic"] == 1)
                    & (df["Churn"] == 0)
                ).sum(),

                (
                    (df["InternetService_Fiber optic"] == 1)
                    & (df["Churn"] == 1)
                ).sum()
            ],

            "No Internet": [
                (
                    (df["InternetService_No"] == 1)
                    & (df["Churn"] == 0)
                ).sum(),

                (
                    (df["InternetService_No"] == 1)
                    & (df["Churn"] == 1)
                ).sum()
            ]
        },
        index=[
            "Stayed",
            "Churned"
        ]
    )

    st.bar_chart(
        internet_df
    )

    st.divider()

    # -----------------------------------------------------
    # PAYMENT METHOD
    # -----------------------------------------------------

    st.subheader(
        "Payment Method and Churn"
    )

    payment_data = pd.DataFrame(
        {
            "Credit Card": [
                (
                    (df["PaymentMethod_Credit card (automatic)"] == 1)
                    & (df["Churn"] == 0)
                ).sum(),

                (
                    (df["PaymentMethod_Credit card (automatic)"] == 1)
                    & (df["Churn"] == 1)
                ).sum()
            ],

            "Electronic Check": [
                (
                    (df["PaymentMethod_Electronic check"] == 1)
                    & (df["Churn"] == 0)
                ).sum(),

                (
                    (df["PaymentMethod_Electronic check"] == 1)
                    & (df["Churn"] == 1)
                ).sum()
            ],

            "Mailed Check": [
                (
                    (df["PaymentMethod_Mailed check"] == 1)
                    & (df["Churn"] == 0)
                ).sum(),

                (
                    (df["PaymentMethod_Mailed check"] == 1)
                    & (df["Churn"] == 1)
                ).sum()
            ]
        },
        index=[
            "Stayed",
            "Churned"
        ]
    )

    st.bar_chart(
        payment_data
    )

    st.divider()

    # -----------------------------------------------------
    # MONTHLY CHARGES
    # -----------------------------------------------------

    st.subheader(
        "Monthly Charges vs Churn"
    )

    monthly_data = (
        df.groupby("Churn")["MonthlyCharges"]
        .mean()
    )

    monthly_display = pd.DataFrame(
        {
            "Average Monthly Charges": [
                monthly_data.get(0, 0),
                monthly_data.get(1, 0)
            ]
        },
        index=[
            "Stayed",
            "Churned"
        ]
    )

    st.bar_chart(
        monthly_display
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.header(
        "🤖 Machine Learning Model Performance"
    )

    if os.path.exists(COMPARISON_PATH):

        comparison = pd.read_csv(
            COMPARISON_PATH
        )

        st.dataframe(
            comparison,
            use_container_width=True
        )

        st.divider()

        st.subheader(
            "Model Metrics"
        )

        metrics = comparison.set_index(
            "Model"
        )

        st.bar_chart(
            metrics[
                [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "ROC-AUC"
                ]
            ]
        )

    else:

        st.warning(
            "Model comparison report not found."
        )


# =========================================================
# CHURN PREDICTION
# =========================================================

elif page == "Churn Prediction":

    st.header(
        "🔮 Customer Churn Prediction"
    )

    st.write(
        "Enter customer information to estimate "
        "the probability of churn."
    )

    col1, col2 = st.columns(2)

    with col1:

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0
        )

        gender_male = st.selectbox(
            "Gender",
            [
                "Female",
                "Male"
            ]
        )

        partner = st.selectbox(
            "Partner",
            [
                "No",
                "Yes"
            ]
        )

        dependents = st.selectbox(
            "Dependents",
            [
                "No",
                "Yes"
            ]
        )

        phone_service = st.selectbox(
            "Phone Service",
            [
                "No",
                "Yes"
            ]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No",
                "Yes",
                "No phone service"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

    with col2:

        online_security = st.selectbox(
            "Online Security",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        device_protection = st.selectbox(
            "Device Protection",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            [
                "No",
                "Yes"
            ]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check"
            ]
        )

    st.divider()

    predict_button = st.button(
        "🔮 Predict Churn",
        use_container_width=True
    )

    if predict_button:

        customer = {
            "SeniorCitizen": 0,

            "tenure": tenure,

            "MonthlyCharges": monthly_charges,

            "TotalCharges": total_charges,

            "gender_Male":
                1 if gender_male == "Male" else 0,

            "Partner_Yes":
                1 if partner == "Yes" else 0,

            "Dependents_Yes":
                1 if dependents == "Yes" else 0,

            "PhoneService_Yes":
                1 if phone_service == "Yes" else 0,

            "MultipleLines_No phone service":
                1 if multiple_lines == "No phone service" else 0,

            "MultipleLines_Yes":
                1 if multiple_lines == "Yes" else 0,

            "InternetService_Fiber optic":
                1 if internet_service == "Fiber optic" else 0,

            "InternetService_No":
                1 if internet_service == "No" else 0,

            "OnlineSecurity_No internet service":
                1 if online_security == "No internet service" else 0,

            "OnlineSecurity_Yes":
                1 if online_security == "Yes" else 0,

            "OnlineBackup_No internet service":
                1 if online_backup == "No internet service" else 0,

            "OnlineBackup_Yes":
                1 if online_backup == "Yes" else 0,

            "DeviceProtection_No internet service":
                1 if device_protection == "No internet service" else 0,

            "DeviceProtection_Yes":
                1 if device_protection == "Yes" else 0,

            "TechSupport_No internet service":
                1 if tech_support == "No internet service" else 0,

            "TechSupport_Yes":
                1 if tech_support == "Yes" else 0,

            "StreamingTV_No internet service":
                1 if streaming_tv == "No internet service" else 0,

            "StreamingTV_Yes":
                1 if streaming_tv == "Yes" else 0,

            "StreamingMovies_No internet service":
                1 if streaming_movies == "No internet service" else 0,

            "StreamingMovies_Yes":
                1 if streaming_movies == "Yes" else 0,

            "Contract_One year":
                1 if contract == "One year" else 0,

            "Contract_Two year":
                1 if contract == "Two year" else 0,

            "PaperlessBilling_Yes":
                1 if paperless_billing == "Yes" else 0,

            "PaymentMethod_Credit card (automatic)":
                1 if payment_method == "Credit card (automatic)" else 0,

            "PaymentMethod_Electronic check":
                1 if payment_method == "Electronic check" else 0,

            "PaymentMethod_Mailed check":
                1 if payment_method == "Mailed check" else 0
        }

        customer_df = pd.DataFrame(
            [customer]
        )

        feature_columns = [
            column
            for column in df.columns
            if column != "Churn"
        ]

        customer_df = customer_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        prediction = model.predict(
            customer_df
        )[0]

        probability = model.predict_proba(
            customer_df
        )[0][1]

        churn_percentage = probability * 100

        if churn_percentage >= 70:

            risk = "HIGH"

        elif churn_percentage >= 40:

            risk = "MEDIUM"

        else:

            risk = "LOW"

        st.divider()

        st.subheader(
            "Prediction Result"
        )

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.metric(
                "Churn Probability",
                f"{churn_percentage:.2f}%"
            )

        with result_col2:

            st.metric(
                "Risk Level",
                risk
            )

        with result_col3:

            if prediction == 1:

                result = "Likely to Churn"

            else:

                result = "Likely to Stay"

            st.metric(
                "Prediction",
                result
            )

        st.progress(
            int(churn_percentage)
        )

        st.divider()

        if risk == "HIGH":

            st.error(
                "⚠️ HIGH RISK: Immediate customer "
                "retention action is recommended."
            )

        elif risk == "MEDIUM":

            st.warning(
                "⚠️ MEDIUM RISK: Customer should "
                "be monitored closely."
            )

        else:

            st.success(
                "✅ LOW RISK: Customer currently has "
                "relatively low churn risk."
            )


# =========================================================
# FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.info(
    "Customer Churn Prediction System\n\n"
    "Built with Python, Pandas, "
    "Scikit-learn and Streamlit."
)