
import streamlit as st
import pandas as pd
import joblib

# Load saved model and feature order
model = joblib.load("heart_disease_model.pkl")
features = joblib.load("heart_disease_features.pkl")

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️"
)

st.title("❤️ Heart Disease Prediction")
st.write(
    "Enter the details below to get a machine-learning prediction."
)

age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
cp = st.selectbox("Chest Pain Type (cp)", [1, 2, 3, 4])
thalach = st.number_input(
    "Maximum Heart Rate Achieved (thalach)",
    min_value=50,
    max_value=250,
    value=150
)

if st.button("Predict"):
    input_data = pd.DataFrame(
        [[age, sex, cp, thalach]],
        columns=features
    )

    prediction = int(model.predict(input_data)[0])

    if prediction == 1:
        st.warning("The model predicts a possible presence of heart disease.")
    else:
        st.success("The model predicts no heart disease according to the selected inputs.")

st.caption(
    "Educational project only. This prediction is not a medical diagnosis."
)
