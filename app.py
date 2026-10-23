import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load("customer_churn_model.pkl")

# Project title
st.title("Customer Churn Prediction")

# Customer information
st.write("Enter Customer Details")

senior = st.selectbox("Senior Citizen", [0, 1])
tenure = st.number_input("Tenure", min_value=0, value=12)
monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total = st.number_input("Total Charges", min_value=0.0, value=840.0)

gender = st.selectbox("Gender", ["Female", "Male"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
phone = st.selectbox("Phone Service", ["Yes", "No"])
multiple = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
device = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"]
)

# Store customer details
data = {
    "SeniorCitizen": senior,
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "gender_Male": int(gender == "Male"),
    "Partner_Yes": int(partner == "Yes"),
    "Dependents_Yes": int(dependents == "Yes"),
    "PhoneService_Yes": int(phone == "Yes"),
    "MultipleLines_No phone service": int(multiple == "No phone service"),
    "MultipleLines_Yes": int(multiple == "Yes"),
    "InternetService_Fiber optic": int(internet == "Fiber optic"),
    "InternetService_No": int(internet == "No"),
    "OnlineSecurity_No internet service": int(security == "No internet service"),
    "OnlineSecurity_Yes": int(security == "Yes"),
    "OnlineBackup_No internet service": int(backup == "No internet service"),
    "OnlineBackup_Yes": int(backup == "Yes"),
    "DeviceProtection_No internet service": int(device == "No internet service"),
    "DeviceProtection_Yes": int(device == "Yes"),
    "TechSupport_No internet service": int(support == "No internet service"),
    "TechSupport_Yes": int(support == "Yes"),
    "StreamingTV_No internet service": int(tv == "No internet service"),
    "StreamingTV_Yes": int(tv == "Yes"),
    "StreamingMovies_No internet service": int(movies == "No internet service"),
    "StreamingMovies_Yes": int(movies == "Yes"),
    "Contract_One year": int(contract == "One year"),
    "Contract_Two year": int(contract == "Two year"),
    "PaperlessBilling_Yes": int(paperless == "Yes"),
    "PaymentMethod_Credit card (automatic)": int(payment == "Credit card (automatic)"),
    "PaymentMethod_Electronic check": int(payment == "Electronic check"),
    "PaymentMethod_Mailed check": int(payment == "Mailed check")
}

# Convert details into a DataFrame
customer = pd.DataFrame([data])

# Predict churn
if st.button("Predict Churn"):
    result = model.predict(customer)

    if result[0] == 1:
        st.write("Customer may leave the company.")
    else:
        st.write("Customer may stay with the company.")