
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(page_title='Salary Predictor Pro', layout='wide')

# Load the model and data for reference
@st.cache_resource
def load_assets():
    model = pickle.load(open('best_salary_prediction_model.pkl', 'rb'))
    data = pd.read_csv('Salary_Data.csv').dropna()
    return model, data

try:
    model, df = load_assets()
except Exception as e:
    st.error("Error loading model or data. Please ensure 'best_salary_prediction_model.pkl' and 'Salary_Data.csv' are in the same folder.")
    st.stop()

# Sidebar
st.sidebar.header('Settings')
st.sidebar.info('This app predicts annual salary based on Age and Years of Experience using a Random Forest Regressor.')

# Main UI
st.title('💵 Professional Salary Prediction Dashboard')

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader('User Input Parameters')
    age = st.slider('Current Age', min_value=18, max_value=70, value=30)
    experience = st.slider('Years of Professional Experience', min_value=0, max_value=50, value=5)
    
    if st.button('Generate Prediction', use_container_width=True):
        features = np.array([[age, experience]])
        prediction = model.predict(features)
        
        st.markdown('---')
        st.metric(label="Estimated Annual Salary", value=f"${prediction[0]:,.2f}")
        st.success('Prediction generated successfully!')

with col2:
    st.subheader('Dataset Insights')
    st.write('Average salaries in our database for your experience level:')
    # Filter data roughly around the user input
    filtered_data = df[(df['Years of Experience'] >= experience - 2) & (df['Years of Experience'] <= experience + 2)]
    if not filtered_data.empty:
        st.line_chart(filtered_data.groupby('Years of Experience')['Salary'].mean())
    else:
        st.write("Not enough data for a comparison chart at this experience level.")

st.markdown('---')
st.caption('Built with Scikit-Learn and Streamlit')
