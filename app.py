import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ----------------------------
# Load Model and Scaler
# ----------------------------

model = load_model("best_lstm_model.keras")
scaler = joblib.load("scaler.pkl")

# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.title("📊 Project Information")

st.sidebar.info("""
Model Used: LSTM

Dataset: Tesla Historical Stock Data

Input: Previous 60 Closing Prices

Output: Next Day Predicted Closing Price
""")

st.sidebar.success("Deep Learning Stock Prediction Project")

# ----------------------------
# Main Title
# ----------------------------

st.title("🚗 Tesla Stock Price Prediction")

st.markdown("""
Predict Tesla's next-day closing stock price using a trained Long Short-Term Memory (LSTM) model.

Enter the last **60 closing prices** separated by commas.
""")

# ----------------------------
# About Project
# ----------------------------

with st.expander("📖 About This Project"):
    st.write("""
    This project predicts Tesla stock prices using Deep Learning.

    Techniques Used:
    - Data Cleaning
    - Exploratory Data Analysis (EDA)
    - Feature Engineering
    - SimpleRNN
    - LSTM
    - Hyperparameter Tuning
    - Streamlit Deployment

    The final deployed model is an optimized LSTM model.
    """)

# ----------------------------
# Sample Data Button
# ----------------------------

sample_data = ",".join(["250"] * 60)

if st.button("Load Sample Data"):
    st.session_state.sample = sample_data

# ----------------------------
# Input Area
# ----------------------------

user_input = st.text_area(
    "Enter 60 Closing Prices",
    value=st.session_state.get("sample", ""),
    height=200
)

# ----------------------------
# Prediction Button
# ----------------------------

if st.button("Predict Stock Price"):

    try:

        values = [float(x.strip()) for x in user_input.split(",")]

        if len(values) != 60:
            st.error(
                f"Please enter exactly 60 values. You entered {len(values)} values."
            )

        else:

            # Input Statistics

            st.subheader("📈 Input Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Minimum Price", f"${min(values):.2f}")

            with col2:
                st.metric("Maximum Price", f"${max(values):.2f}")

            with col3:
                st.metric(
                    "Average Price",
                    f"${sum(values)/len(values):.2f}"
                )

            # Trend Chart

            st.subheader("📉 Last 60 Days Trend")

            st.line_chart(values)

            # Preprocessing

            data = np.array(values).reshape(-1, 1)

            scaled = scaler.transform(data)

            X = scaled.reshape(1, 60, 1)

            # Prediction

            with st.spinner("Predicting stock price..."):

                prediction = model.predict(X)

            prediction = scaler.inverse_transform(prediction)

            predicted_price = prediction[0][0]

            # Results

            st.subheader("🎯 Prediction Result")

            st.metric(
                label="Predicted Tesla Closing Price",
                value=f"${predicted_price:.2f}"
            )

            st.info(
                f"Estimated Price Range: "
                f"${predicted_price-5:.2f} to "
                f"${predicted_price+5:.2f}"
            )

            st.success("Prediction completed successfully!")

    except Exception as e:
        st.error(f"Error: {e}")

# ----------------------------
# Footer
# ----------------------------

st.markdown("---")

st.markdown(
    """
    **Developed by Shivani Chaudhary**  
    B.Tech AIML | Tesla Stock Price Prediction using LSTM
    """
)


           
