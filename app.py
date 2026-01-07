import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="📊",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
body {
    background-color: white;
}
.big-title {
    font-size: 42px;
    font-weight: 800;
    color: #0A4D68;
}
.sub-title {
    font-size: 18px;
    color: #4F709C;
}
.metric-box {
    background-color: #F7F9FB;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.markdown('<div class="big-title">📊 Telco Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Professional ML Dashboard with Dynamic Simulation</div>', unsafe_allow_html=True)
st.markdown("---")

# -------------------- LOAD & PREPROCESS DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df, drop_first=True)
    return df

df = load_data()

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------- SIDEBAR INPUTS --------------------
st.sidebar.header("🧑 Customer Simulation")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly = st.sidebar.slider("Monthly Charges", 20.0, 120.0, 70.0)
total = st.sidebar.slider("Total Charges", 20.0, 9000.0, 2000.0)

threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.5)

# -------------------- BUILD CUSTOMER INPUT --------------------
customer = pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns)

customer['tenure'] = tenure
customer['MonthlyCharges'] = monthly
customer['TotalCharges'] = total

if gender == "Male":
    customer['gender_Male'] = 1
if senior == "Yes":
    customer['SeniorCitizen'] = 1
if partner == "Yes":
    customer['Partner_Yes'] = 1
if dependents == "Yes":
    customer['Dependents_Yes'] = 1
if contract == "One year":
    customer['Contract_One year'] = 1
elif contract == "Two year":
    customer['Contract_Two year'] = 1

customer_scaled = scaler.transform(customer)

# -------------------- PREDICTION --------------------
churn_prob = model.predict_proba(customer_scaled)[0][1]
prediction = int(churn_prob >= threshold)

# -------------------- DISPLAY PREDICTION --------------------
st.subheader("🔮 Customer Churn Prediction")

col1, col2, col3 = st.columns(3)

col1.metric("Churn Probability", f"{churn_prob:.2f}")
col2.metric("Threshold", f"{threshold:.2f}")
col3.metric("Prediction", "Churn" if prediction else "No Churn")


# -------------------- MODEL PERFORMANCE --------------------
y_probs = model.predict_proba(X_test)[:, 1]
y_pred = (y_probs >= threshold).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.markdown("---")
st.subheader("📈 Model Performance (Test Data)")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{acc:.2f}")
m2.metric("Precision", f"{prec:.2f}")
m3.metric("Recall", f"{rec:.2f}")
m4.metric("F1 Score", f"{f1:.2f}")

st.info(
    "Confusion matrix represents overall model performance on test data. "
    "Customer simulation affects only individual prediction."
)

# -------------------- CONFUSION MATRIX --------------------
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    ax=ax
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix", fontsize=12)

st.pyplot(fig)

# -------------------- MODEL INFO --------------------
with st.expander("🔍 Model Details"):
    st.markdown("""
    **Algorithm:** Logistic Regression  
    **Dataset:** Telco Customer Churn  
    **Features:** Demographic, Contract, Billing  
    **Goal:** Predict churn probability for early retention
    """)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Scikit-learn | Professional ML Dashboard")