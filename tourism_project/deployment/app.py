import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="Surendra2025/Model_repo", filename="best_package_model.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism App")
st.write("""
This application predicts potential buyers, and enhances decision-making for marketing strategies.
Please enter the sensor and configuration data below to get a prediction.
""")

# User input
gender = st.selectbox("Gender", ["Male", "Female", "Fe Male"])
status = st.selectbox("MaritalStatus", ["Single", "Unmarried"])
Occu = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business"])
designation = st.selectbox("Designation", ["AVP", "Executive", "Manager", "Senior Manager"])
age = st.number_input("Age", min_value=26, max_value=60, value=38)
income = st.number_input("MonthlyIncome", min_value=23500, max_value=28600, value=25500)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'MonthlyIncome': income,
    'Gender': gender,
    'MaritalStatus': status,
    'Occupation': Occu,
    'Designation': designation
}])


if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Purchase" if prediction == 1 else "No Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
