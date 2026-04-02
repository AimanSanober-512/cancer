import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Set Page Config
st.set_page_config(
    page_title="Lung Cancer Detector",
    page_icon="🫁",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #2c3e50;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #34495e;
        border: 1px solid #2c3e50;
    }
    .title-container {
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #2c3e50;
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 25px;
    }
    .result-container {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
    .high-risk {
        background-color: #ffcccc;
        border: 2px solid #ff0000;
        color: #b30000;
    }
    .low-risk {
        background-color: #ccffcc;
        border: 2px solid #009900;
        color: #006600;
    }
</style>
""", unsafe_allow_html=True)

# Load Model and Preprocessing Objects
@st.cache_resource
def load_assets():
    model = joblib.load('rfc_model.pkl') # Renamed to match the one I'll generate
    scaler = joblib.load('scaler.pkl')
    le_gender = joblib.load('le_gender.pkl')
    le_target = joblib.load('le_target.pkl')
    return model, scaler, le_gender, le_target

# Header
st.markdown('<div class="title-container"><h1>🫁 Lung Cancer Prediction System</h1></div>', unsafe_allow_html=True)
st.write("Please fill in the patient details below to assess the risk of lung cancer.")

# Main Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics & Habits")
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        gender = st.selectbox("Gender", options=["Male", "Female"])
        smoking = st.radio("Smoking Habit", options=["No", "Yes"])
        alcohol = st.radio("Alcohol Consuming", options=["No", "Yes"])
        peer_pressure = st.radio("Peer Pressure", options=["No", "Yes"])
        anxiety = st.radio("Anxiety", options=["No", "Yes"])

    with col2:
        st.subheader("Symptoms & Health Info")
        yellow_fingers = st.radio("Yellow Fingers", options=["No", "Yes"])
        chronic_disease = st.radio("Chronic Disease", options=["No", "Yes"])
        fatigue = st.radio("Fatigue", options=["No", "Yes"])
        allergy = st.radio("Allergy", options=["No", "Yes"])
        wheezing = st.radio("Wheezing", options=["No", "Yes"])
        coughing = st.radio("Coughing", options=["No", "Yes"])
        shortness_of_breath = st.radio("Shortness of Breath", options=["No", "Yes"])
        swallowing_difficulty = st.radio("Swallowing Difficulty", options=["No", "Yes"])
        chest_pain = st.radio("Chest Pain", options=["No", "Yes"])

    submit_button = st.form_submit_button("Predict Result")

if submit_button:
    try:
        # Load assets (Ensure they exist)
        try:
            model = joblib.load('model.pkl')
            scaler = joblib.load('scaler.pkl')
            le_gender = joblib.load('le_gender.pkl')
        except:
            st.error("Error: Model files not found. Please run the training notebook first.")
            st.stop()

        # Map inputs to model format
        # NO:0, YES:1
        def map_choice(c): return 1 if c == "Yes" else 0
        def map_gender(g): return 1 if g == "Female" else 0
        
        # Original columns: GENDER,AGE,SMOKING,YELLOW_FINGERS,ANXIETY,PEER_PRESSURE,CHRONIC DISEASE,FATIGUE,ALLERGY,WHEEZING,ALCOHOL CONSUMING,COUGHING,SHORTNESS OF BREATH,SWALLOWING DIFFICULTY,CHEST PAIN
        input_data = pd.DataFrame([{
            'GENDER': map_gender(gender),
            'AGE': age,
            'SMOKING': map_choice(smoking),
            'YELLOW_FINGERS': map_choice(yellow_fingers),
            'ANXIETY': map_choice(anxiety),
            'PEER_PRESSURE': map_choice(peer_pressure),
            'CHRONIC DISEASE': map_choice(chronic_disease),
            'FATIGUE ': map_choice(fatigue),
            'ALLERGY ': map_choice(allergy),
            'WHEEZING': map_choice(wheezing),
            'ALCOHOL CONSUMING': map_choice(alcohol),
            'COUGHING': map_choice(coughing),
            'SHORTNESS OF BREATH': map_choice(shortness_of_breath),
            'SWALLOWING DIFFICULTY': map_choice(swallowing_difficulty),
            'CHEST PAIN': map_choice(chest_pain)
        }])

        # Scale Age
        input_data[['AGE']] = scaler.transform(input_data[['AGE']])

        # Prediction
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        # Display Result
        if prediction == 1:
            st.markdown(f"""
            <div class="result-container high-risk">
                <h2>Prediction: LUNG CANCER DETECTED (YES)</h2>
                <p>Confidence Level: {prob:.2%}</p>
                <p>Please consult a medical professional immediately for further diagnosis.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-container low-risk">
                <h2>Prediction: NO LUNG CANCER DETECTED</h2>
                <p>Confidence Level: {1-prob:.2%}</p>
                <p>Regular checkups are still recommended for maintaining health.</p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.markdown("© 2026 Lung Cancer Detection System | For Educational Purposes Only")
