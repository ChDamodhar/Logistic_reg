import streamlit as st
import numpy as np
import joblib

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📊 Customer Churn Prediction")

monthly = st.number_input("Monthly Charges", min_value=0.0)
total = st.number_input("Total Charges", min_value=0.0)

if st.button("Predict"):
    X = np.array([[monthly, total]])
    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[0][1]
    pred = model.predict(X_scaled)[0]

    st.write(f"Churn Probability: {prob:.2f}")

    if pred == 1:
        st.error("⚠️ Customer likely to churn")
    else:
        st.success("✅ Customer likely to stay")
