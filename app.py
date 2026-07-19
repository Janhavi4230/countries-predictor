import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load("model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_names = joblib.load("feature_names.pkl")
df = pd.read_csv("Countries.csv")

# Page Configuration
st.set_page_config(
    page_title="Countries Life Expectancy Predictor",
    page_icon="🌍",
    layout="centered"
)

# Title
st.title("🌍 Countries Life Expectancy Predictor")
st.write("Enter country details to predict Life Expectancy.")

# Create input dictionary
input_data = {}

for col in feature_names:

    # st.selectbox()
    if col in label_encoders:
        options = list(label_encoders[col].classes_)
        value = st.selectbox(
            col.replace("_", " ").title(),
            options
        )
        input_data[col] = label_encoders[col].transform([value])[0]

    # st.number_input()
    else:
        value = st.number_input(
            col.replace("_", " ").title(),
            value=float(df[col].median())
        )
        input_data[col] = value

# st.button()
if st.button("🚀 Predict Life Expectancy"):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)

    # Prediction Successfully
    st.success(
        f"🌍 Predicted Life Expectancy : {prediction[0]:.2f} Years"
    )

    st.balloons()