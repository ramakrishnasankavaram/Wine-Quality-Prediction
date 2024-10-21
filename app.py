import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
adaboost_model = pickle.load(open('models/ada.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Page header
st.title("🍷 Wine Quality Prediction")
st.write("Predict the quality of wine based on various chemical properties.")

# Create columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    fixed_acidity = st.text_input('Fixed Acidity', placeholder='e.g., 7.4')
    residual_sugar = st.text_input('Residual Sugar', placeholder='e.g., 1.9')
    total_sulfur_dioxide = st.text_input('Total Sulfur Dioxide', placeholder='e.g., 34')


with col2:
    volatile_acidity = st.text_input('Volatile Acidity', placeholder='e.g., 0.70')
    chlorides = st.text_input('Chlorides', placeholder='e.g., 0.076')
    density = st.text_input('Density', placeholder='e.g., 0.9978')
with col3:
    citric_acid = st.text_input('Citric Acid', placeholder='e.g., 0.00')
    free_sulfur_dioxide = st.text_input('Free Sulfur Dioxide', placeholder='e.g., 11')
    pH = st.text_input('pH', placeholder='e.g., 3.51')

# Row for sulphates and alcohol
sulphates = st.text_input('Sulphates', placeholder='e.g., 0.56')
alcohol = st.text_input('Alcohol', placeholder='e.g., 9.4')

# Check if all inputs are filled
if st.button('Predict'):
    # Validate inputs and convert them to proper data types
        try:
            # Convert inputs to floats
            input_data = [[
                float(fixed_acidity),
                float(volatile_acidity),
                float(citric_acid),
                float(residual_sugar),
                float(chlorides),
                float(free_sulfur_dioxide),
                float(total_sulfur_dioxide),
                float(density),
                float(pH),
                float(sulphates),
                float(alcohol)
            ]]

            # Scale the input data
            scaled_data = standard_scaler.transform(input_data)

            # Predict using the model
            predicted_data = adaboost_model.predict(scaled_data)

            st.success("Prediction completed!")

            st.subheader("Prediction Result")
            if predicted_data == 0:
                st.error("⚠️ Bad Quality Wine")
            else:
                st.success("✅ Good Quality Wine")

        except ValueError:
            st.error("Please enter valid numbers for all input fields.")
