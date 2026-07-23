import streamlit as st  # 1. FIXED: Changed 'as pd' to 'as st'
import pandas as pd
import pickle 

# 1. Page Configuration
st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("Student Performance Predictor")
st.write("Enter your study habits below to predict your score.")

# 2. Load Model Safely 
@st.cache_resource
def load_model():
    with open('predictions.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found! Please place 'predictions.pkl' in the same directory.")
    st.stop()

# 3. Create Input Fields in Sidebar or Main Page
st.header("Input Features")

hours_studied = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=58, step=1)

# Extracurriculars: User chooses Yes/No, code converts to 1/0
extracurricular_input = st.selectbox("Extracurricular Activities", ["Yes", "No"])
extracurriculars = 1 if extracurricular_input == "Yes" else 0

sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=8.0, step=0.5)
papers_practiced = st.number_input("Sample Question Papers Practiced", min_value=0, max_value=50, value=2, step=1)

# 4. Predict Button and Layout
if st.button("Predict Score", type="primary"):
    # 2. FIXED: Fixed closing brackets, added missing comma, and fixed capitalization of papers_practiced
    input_data = pd.DataFrame([[
        hours_studied, 
        previous_scores, 
        extracurriculars, 
        sleep_hours, 
        papers_practiced
    ]], 
    columns=[
        'Hours Studied', 
        'Previous Scores', 
        'Extracurricular Activities', 
        'Sleep Hours', 
        'Sample Question Papers Practiced'
    ])
    
    # Generate prediction
    predictions = model.predict(input_data)
    
    # Display output nicely
    # 3. FIXED: Handles prediction display safely if model returns a single number or an array
    val = predictions[0] if hasattr(predictions, "__len__") else predictions
    st.success(f"Predicted Output: **{val:.3f}**")
# Student Performance Multiple Linear Regression