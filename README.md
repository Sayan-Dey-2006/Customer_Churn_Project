# 📊 Customer Churn Prediction & Analysis

A complete Machine Learning and Data Analytics project designed to identify customers who are likely to leave a telecom service.

The project performs data cleaning, exploratory data analysis, feature engineering, machine learning model training, model comparison, churn prediction, and interactive visualization through a Streamlit dashboard.

---

## 🎯 Project Objective

Customer churn is a major business problem for subscription-based companies.

The objective of this project is to:

* Analyze customer behavior
* Identify important factors affecting churn
* Predict whether a customer is likely to churn
* Identify high-risk customers
* Compare different Machine Learning models
* Provide business recommendations for customer retention

---

## 📂 Project Structure

```text
Customer_Churn_Project/
│
├── data/
│   ├── churn.csv
│   └── processed_churn.csv
│
├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   └── random_forest_model.pkl
│
├── reports/
│   ├── model_comparison.csv
│   ├── model_evaluation.csv
│   └── random_forest_feature_importance.csv
│
├── visualizations/
│   ├── churn_distribution.png
│   ├── churn_by_contract.png
│   ├── churn_by_gender.png
│   ├── churn_by_internet_service.png
│   ├── churn_by_payment_method.png
│   ├── churn_by_senior_citizen.png
│   ├── churn_by_partner.png
│   ├── churn_by_dependents.png
│   ├── monthly_charges_vs_churn.png
│   └── tenure_vs_churn.png
│
├── src/
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── random_forest.py
│   ├── model_comparison.py
│   ├── model_evaluation.py
│   ├── predict.py
│   └── dashboard.py
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib
* Streamlit

### Machine Learning

* Logistic Regression
* Random Forest Classifier

### Visualization

* Matplotlib
* Streamlit

---

## 📊 Dataset

The dataset contains customer-level telecom information.

### Dataset Size

* Original Rows: **7,043**
* Original Columns: **21**
* Final processed columns: **31**

### Target Variable

```text
Churn
```

Where:

```text
0 = Customer stayed
1 = Customer churned
```

---

## 🔍 Data Cleaning

The data cleaning stage performs:

* Dataset validation
* Data type inspection
* Missing value detection
* Duplicate detection
* Churn distribution analysis
* Conversion of `TotalCharges` into numeric format

The original dataset contained **11 missing values in TotalCharges** after numeric conversion.

These values were handled during preprocessing.

---

## 📈 Exploratory Data Analysis

The EDA stage analyzes customer churn across different customer characteristics.

### Churn Rate

The dataset contains:

* Total customers after cleaning: **7,032**
* Customers who stayed: **5,163**
* Customers who churned: **1,869**
* Overall churn rate: **26.58%**

### Important EDA Findings

#### Contract

Month-to-month customers showed significantly higher churn compared with customers on one-year or two-year contracts.

#### Internet Service

Fiber optic customers showed considerably higher churn than DSL and customers without internet service.

#### Payment Method

Electronic check customers showed a noticeably higher number of churned customers.

#### Senior Citizens

Senior citizens had a higher churn proportion compared with non-senior customers.

#### Dependents

Customers without dependents showed higher churn than customers with dependents.

#### Partner

Customers without a partner showed higher churn than customers with a partner.

---

## ⚙️ Feature Engineering

The feature engineering pipeline performs:

* Removal of `customerID`
* Conversion of `TotalCharges` to numeric
* Missing value handling
* Conversion of `Churn` into binary values
* One-hot encoding of categorical variables
* Feature preparation for Machine Learning

Final processed dataset:

```text
Rows    : 7043
Columns : 31
Missing : 0
```

---

# 🤖 Machine Learning

Two classification models were trained and evaluated.

## 1. Logistic Regression

Logistic Regression was used as the primary classification model.

### Performance

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 80.70% |
| Precision | 65.84% |
| Recall    | 56.68% |
| F1 Score  | 60.92% |
| ROC-AUC   | 84.16% |

---

## 2. Random Forest

Random Forest was trained to capture more complex relationships between customer characteristics and churn.

### Performance

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 75.87% |
| Precision | 53.18% |
| Recall    | 75.94% |
| F1 Score  | 62.56% |
| ROC-AUC   | 84.13% |

---

# 📊 Model Comparison

| Model               | Accuracy | Precision |     Recall |   F1 Score |    ROC-AUC |
| ------------------- | -------: | --------: | ---------: | ---------: | ---------: |
| Logistic Regression |   80.70% |    65.84% |     56.68% |     60.92% | **84.16%** |
| Random Forest       |   75.87% |    53.18% | **75.94%** | **62.56%** |     84.13% |

### Best Model by Metric

* **Accuracy:** Logistic Regression
* **Precision:** Logistic Regression
* **Recall:** Random Forest
* **F1 Score:** Random Forest
* **ROC-AUC:** Logistic Regression

### Model Selection

For general balanced performance, **Logistic Regression** performed better in accuracy, precision, and ROC-AUC.

For identifying more potential churn customers, **Random Forest** is useful because it achieved a significantly higher recall.

---

# 🔎 Important Churn Factors

The models identified several important features associated with churn.

### Logistic Regression

Important features included:

* Tenure
* Monthly Charges
* Fiber Optic Internet Service
* Two-Year Contract
* Total Charges
* One-Year Contract
* Streaming Movies
* Streaming TV
* Multiple Lines
* Paperless Billing
* Electronic Check Payment

### Random Forest

Important features included:

* Tenure
* Total Charges
* Monthly Charges
* Two-Year Contract
* Fiber Optic Internet Service
* Electronic Check
* One-Year Contract
* Online Security
* Tech Support
* Paperless Billing

---

# 🔮 Customer Churn Prediction

The project includes an interactive prediction system.

Users can enter customer information such as:

* Tenure
* Monthly Charges
* Total Charges
* Gender
* Partner
* Dependents
* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies
* Contract
* Paperless Billing
* Payment Method

The system returns:

```text
Churn Probability
Risk Level
Prediction
Business Recommendation
```

Example:

```text
Churn Probability : 74.62%
Risk Level        : HIGH
Prediction        : Customer is likely to churn.
```

---

# 📊 Streamlit Dashboard

The project also includes an interactive Streamlit dashboard.

The dashboard provides a visual overview of:

* Customer churn distribution
* Churn rate
* Customer demographics
* Contract analysis
* Internet service analysis
* Payment method analysis
* Model performance
* Churn prediction

### Run Dashboard

Open the terminal inside the project folder and run:

```bash
streamlit run src/dashboard.py
```

The dashboard will open in the browser.

---

# 💼 Business Recommendations

Based on the analysis, businesses can focus on:

### 1. Month-to-Month Customers

Offer incentives to encourage customers to move to longer-term contracts.

### 2. High Monthly Charges

Identify customers with high monthly charges and provide personalized plans or benefits.

### 3. Fiber Optic Customers

Investigate service quality, pricing, and customer satisfaction among fiber optic users.

### 4. Electronic Check Customers

Analyze whether payment experience or billing issues are contributing to churn.

### 5. New Customers

Customers with shorter tenure should receive additional onboarding and support.

### 6. High-Risk Customers

Use the prediction system to identify customers with high churn probability and prioritize retention campaigns.

---

# 🚀 How to Run the Project

## Step 1 — Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

## Step 2 — Open the Project

```bash
cd Customer_Churn_Project
```

## Step 3 — Create Virtual Environment

```bash
python -m venv .venv
```

## Step 4 — Activate Virtual Environment

### Windows

```bash
.venv\Scripts\activate
```

## Step 5 — Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 6 — Run Data Cleaning

```bash
python src/data_cleaning.py
```

## Step 7 — Run EDA

```bash
python src/eda.py
```

## Step 8 — Run Feature Engineering

```bash
python src/feature_engineering.py
```

## Step 9 — Train Logistic Regression

```bash
python src/model.py
```

## Step 10 — Train Random Forest

```bash
python src/random_forest.py
```

## Step 11 — Compare Models

```bash
python src/model_comparison.py
```

## Step 12 — Evaluate Models

```bash
python src/model_evaluation.py
```

## Step 13 — Run Prediction System

```bash
python src/predict.py
```

## Step 14 — Launch Dashboard

```bash
streamlit run src/dashboard.py
```

---

# 📌 Key Results

```text
Total Customers          : 7,043
Churn Rate               : 26.58%

Logistic Regression
Accuracy                 : 80.70%
ROC-AUC                  : 84.16%

Random Forest
Accuracy                 : 75.87%
Recall                   : 75.94%
ROC-AUC                  : 84.13%
```

---

# 🎓 Project Learning Outcomes

Through this project, I practiced:

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Categorical Encoding
* Machine Learning
* Classification
* Model Evaluation
* Confusion Matrix
* ROC-AUC Analysis
* Feature Importance
* Customer Risk Prediction
* Business Analysis
* Streamlit Dashboard Development

---

# 👨‍💻 Author

**Sayan Dey**

BCA Student | Aspiring Data Analyst

Interested in:

* Data Analytics
* Machine Learning
* Python
* Business Intelligence
* Real-world Problem Solving

---

## ⭐ Project Status

**Completed**

The project includes the complete pipeline from raw customer data to Machine Learning prediction and an interactive business dashboard.
