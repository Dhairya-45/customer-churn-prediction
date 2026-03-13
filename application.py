import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.title("Customer Churn Prediction")

# Load model and scaler
@st.cache_resource
def load_artifacts():
    with open("pkl/churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("pkl/churn_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}. Run notebook cells 39-40 first.")
    st.stop()

st.header("Single Customer Prediction")

invoice  = st.number_input("Invoice (Purchase Frequency)", min_value=1, value=5)
quantity = st.number_input("Quantity (Total Items)", min_value=1, value=50)
price    = st.number_input("Price (Total Spend £)", min_value=0.01, value=200.0)

if st.button("Predict"):
    input_df = pd.DataFrame([[invoice, quantity, price]], columns=["Invoice", "Quantity", "Price"])
    X = scaler.transform(input_df)
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]

    if pred == 1:
        st.error(f"⚠️ Customer is likely to CHURN — Probability: {proba:.2%}")
    else:
        st.success(f"✅ Customer is likely to be RETAINED — Churn Probability: {proba:.2%}")

st.divider()

st.header("Batch Prediction")
st.write("Upload a CSV with columns: Invoice, Quantity, Price")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    batch = pd.read_csv(uploaded)
    X_batch = scaler.transform(batch[["Invoice", "Quantity", "Price"]])
    batch["ChurnProbability"] = np.round(model.predict_proba(X_batch)[:, 1], 4)
    batch["Prediction"]       = model.predict(X_batch)
    batch["Result"]           = batch["Prediction"].map({0: "Retained", 1: "Churn"})

    st.write(f"Total: {len(batch)} | Churners: {batch['Prediction'].sum()} | Rate: {batch['Prediction'].mean():.1%}")
    st.dataframe(batch.sort_values("ChurnProbability", ascending=False))

    st.download_button("Download Results", batch.to_csv(index=False), "churn_results.csv", "text/csv")