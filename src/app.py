import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Boston House Price Prediction", 
    page_icon="🏠", 
    layout="wide"
)

@st.cache_resource
def load_model():
    try:
        model = joblib.load('trained_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        with open('feature_names.txt', 'r') as f:
            feature_names = f.read().splitlines()
        
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def main():
    st.title("🏠 Boston House Price Prediction")
    st.write("Enter the following details to predict the house price:")

    model, scaler, feature_names = load_model()
    if model is None or scaler is None or feature_names is None:
        st.error("Could not load the model or associated files. Please check your setup.")
        return

    selected_features = ['CRIM', 'RM', 'DIS', 'LSTAT', 'PTRATIO']

    user_input = {}
    for feature_name in selected_features:
        user_input[feature_name] = st.number_input(
            f"Enter value for {feature_name}:", value=0.0, step=0.01
        )

    if st.button("Predict House Price"):
        try:
            input_data = pd.DataFrame([user_input])

            for feature in feature_names:
                if feature not in input_data.columns:
                    input_data[feature] = 0.0 

            input_data = input_data[feature_names]

            input_scaled = scaler.transform(input_data)

            prediction = model.predict(input_scaled)
            predicted_price = prediction[0]
            
            st.subheader("Prediction Results")
            st.metric(label="House Price (MEDV)", value=f"${predicted_price:,.2f}")

        except Exception as e:
            st.error(f"Prediction error: {e}")

if __name__ == '__main__':
    main()
