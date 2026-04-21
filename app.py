
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model
model = pickle.load(open('best_salary_prediction_model.pkl', 'rb'))

st.title('Salary Prediction App')

age = st.number_input('Age', min_value=18, max_value=100, value=30)
experience = st.number_input('Years of Experience', min_value=0, max_value=50, value=5)

if st.button('Predict Salary'):
    features = np.array([[age, experience]])
    prediction = model.predict(features)
    st.success(f'Estimated Salary: ${prediction[0]:,.2f}')
