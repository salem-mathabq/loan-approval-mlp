import json, os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import joblib

OUT = "outputs"
MODEL_PATH = os.path.join(OUT, "best_model.keras")
SCALER_PATH = os.path.join(OUT, "scaler.joblib")
FEATURES_PATH = os.path.join(OUT, "feature_columns.json")

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦")
st.title("Loan Approval Prediction")
st.caption("Educational demonstration only — not a real lending decision.")

if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]):
    st.error("Model artifacts not found. Run `python solution.py` first.")
    st.stop()

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    return model, scaler, features

model, scaler, features = load_artifacts()

with st.form("loan_form"):
    dependents = st.number_input("Number of dependents", min_value=0, max_value=20, value=0, step=1)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self-employed", ["No", "Yes"])
    income = st.number_input("Annual income", min_value=0, value=500000, step=50000)
    loan_amount = st.number_input("Loan amount", min_value=0, value=1000000, step=50000)
    term = st.number_input("Loan term", min_value=1, max_value=100, value=10, step=1)
    cibil = st.number_input("CIBIL score", min_value=0, max_value=900, value=700, step=1)
    residential = st.number_input("Residential assets value", min_value=0, value=1000000, step=50000)
    commercial = st.number_input("Commercial assets value", min_value=0, value=0, step=50000)
    luxury = st.number_input("Luxury assets value", min_value=0, value=500000, step=50000)
    bank = st.number_input("Bank asset value", min_value=0, value=200000, step=10000)
    submitted = st.form_submit_button("Predict")

if submitted:
    values = {
        "no_of_dependents": dependents,
        "education": 1 if education == "Graduate" else 0,
        "self_employed": 1 if self_employed == "Yes" else 0,
        "income_annum": income,
        "loan_amount": loan_amount,
        "loan_term": term,
        "cibil_score": cibil,
        "residential_assets_value": residential,
        "commercial_assets_value": commercial,
        "luxury_assets_value": luxury,
        "bank_asset_value": bank,
    }
    row = pd.DataFrame([[values[c] for c in features]], columns=features)
    x = scaler.transform(row).astype("float32")
    prob = float(model.predict(x, verbose=0)[0,0])
    pred = "Approved" if prob >= 0.5 else "Rejected"
    st.subheader(f"Predicted class: {pred}")
    st.write(f"Model probability of approval: **{prob:.1%}**")
    st.progress(min(max(prob, 0.0), 1.0))
    st.warning("This prediction reflects patterns in historical data and may be biased. It must not be used as the sole basis for a real credit decision.")
