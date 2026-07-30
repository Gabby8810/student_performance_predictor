import streamlit as st
import pandas as pd
import pickle



st.set_page_config(
    page_title="Student Performance Predictor",
    layout="centered"
)

st.title("Student Performance Predictor")

st.markdown("""
Predict a student's expected academic performance using a **Multiple Linear Regression** model trained on historical student data.

### Input Features
- Hours Studied
- Previous Scores
- Extracurricular Activities
- Sleep Hours
- Sample Question Papers Practiced
""")



@st.cache_resource
def load_model():
    with open("predictions.pkl", "rb") as file:
        return pickle.load(file)

try:
    model = load_model()

except FileNotFoundError:
    st.error("Model file 'predictions.pkl' was not found.")
    st.stop()


st.subheader("Enter Student Information")

hours_studied = st.number_input(
    "Hours Studied",
    min_value=0.0,
    max_value=24.0,
    value=6.0,
    step=0.5
)

previous_scores = st.number_input(
    "Previous Scores",
    min_value=0,
    max_value=100,
    value=58
)

activity = st.selectbox(
    "Extracurricular Activities",
    ["Yes", "No"]
)

extracurriculars = 1 if activity == "Yes" else 0

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=0.0,
    max_value=24.0,
    value=8.0,
    step=0.5
)

papers_practiced = st.number_input(
    "Sample Question Papers Practiced",
    min_value=0,
    max_value=50,
    value=2
)


if st.button("Predict Score", type="primary"):

    input_data = pd.DataFrame(
        [[
            hours_studied,
            previous_scores,
            extracurriculars,
            sleep_hours,
            papers_practiced
        ]],
        columns=[
            "Hours Studied",
            "Previous Scores",
            "Extracurricular Activities",
            "Sleep Hours",
            "Sample Question Papers Practiced"
        ]
    )

    # Generate prediction
    prediction = model.predict(input_data)

    score = prediction[0]

    # Keep score within 0–100
    score = max(0, min(100, score))

    st.divider()

   

    st.metric(
        label="Predicted Academic Performance",
        value=f"{score:.2f}/100"
    )

    st.progress(score / 100)

    # Performance Level

    if score >= 80:
        st.success("Performance Level: Excellent")

    elif score >= 65:
        st.info("Performance Level: Good")

    elif score >= 50:
        st.warning("Performance Level: Average")

    else:
        st.error("Performance Level: Needs Improvement")

    
    st.caption(
        "This prediction is an estimate based on historical student performance data. Average prediction error is ±1.61 marks" 
    )