import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fraud Prediction App", layout="centered")

st.title("Fraud Prediction App")

# Load model & preprocessor
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Column definitions
numerical_cols = ["amount", "customer_age", "previous_transactions"]
categorical_cols = ["merchant_category", "customer_location", "device_type"]

# Dropdown categories from your dataset
LOCATIONS = ["NY", "IL", "TX", "FL", "CA"]
DEVICES = ["tablet", "mobile", "desktop"]
MERCHANTS = ["fuel", "electronics", "entertainment", "fashion", "grocery"]

st.header("Enter Transaction Details")

with st.form("fraud_form"):
    amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
    merchant = st.selectbox("Merchant Category", MERCHANTS)
    age = st.number_input("Customer Age", min_value=1, max_value=120, value=30)
    location = st.selectbox("Customer Location", LOCATIONS)
    device = st.selectbox("Device Type", DEVICES)
    prev = st.number_input("Previous Transactions", min_value=0, value=1)

    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([{
        "amount": amount,
        "merchant_category": merchant,
        "customer_age": age,
        "customer_location": location,
        "device_type": device,
        "previous_transactions": prev
    }])

    # Preprocess
    Xp = preprocessor.transform(input_df)

    # Predict
    pred = model.predict(Xp)[0]
    prob = model.predict_proba(Xp)[0][1]

    st.subheader("Prediction Result")
    st.write("### 💡 **Fraud**" if pred == 1 else "### ✅ **Not Fraud**")
    st.write(f"**Probability:** {prob:.2%}")
