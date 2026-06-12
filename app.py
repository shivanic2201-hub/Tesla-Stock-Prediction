import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("best_lstm_model.keras")
scaler = joblib.load("scaler.pkl")

st.title("Tesla Stock Price Prediction")

user_input = st.text_area(
    "Enter last 60 closing prices separated by commas"
)

if st.button("Predict"):

    try:
        values = [float(x.strip()) for x in user_input.split(",")]

        if len(values) != 60:
            st.error("Please enter exactly 60 values")

        else:
            data = np.array(values).reshape(-1,1)

            scaled = scaler.transform(data)

            X = scaled.reshape(1,60,1)

            prediction = model.predict(X)

            prediction = scaler.inverse_transform(prediction)

            st.success(
                f"Predicted Closing Price: ${prediction[0][0]:.2f}"
            )

    except Exception as e:
        st.error(f"Error: {e}")
