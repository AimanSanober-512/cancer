import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set Page Config
st.set_page_config(
    page_title="Multi-Cancer Detection System",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for Premium Unified Look
st.markdown("""
<style>
    /* Main Background */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* Modern Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
        box-shadow: 5px 0 15px rgba(0,0,0,0.3);
    }

    /* Sidebar Content Color */
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] div[role="radiogroup"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 1.1rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }

    /* Sidebar Title */
    section[data-testid="stSidebar"] h2 {
        color: #3b82f6 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    /* Glassmorphism for About Box */
    .info-box {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #f8fafc !important;
        margin-top: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Radio Selection Styling */
    div[role="radiogroup"] > label:hover {
        background: rgba(59, 130, 246, 0.2);
        border-radius: 8px;
        transition: 0.3s ease;
    }

    /* Header Container */
    .header-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
        padding: 40px;
        border-radius: 20px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        text-align: center;
    }

    .header-container h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    /* Card Styling */
    .stForm {
        background-color: white;
        padding: 35px;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.05);
        margin-bottom: 25px;
        border-top: 5px solid #1e293b;
    }

    /* Result Container */
    .result-container {
        padding: 30px;
        border-radius: 15px;
        margin-top: 30px;
        text-align: center;
        animation: fadeIn 0.8s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .high-risk {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        border-left: 10px solid #ff4b4b;
        color: #900;
    }

    .low-risk {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        border-left: 10px solid #2ecc71;
        color: #006400;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 50px;
        height: 4rem;
        background: linear-gradient(90deg, #1a1c2c 0%, #4a192c 100%);
        color: white !important;
        font-size: 1.2rem !important;
        font-weight: bold;
        transition: all 0.4s ease;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        border: 2px solid white !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper for loading assets
@st.cache_resource
def load_skin_assets():
    try:
        model = joblib.load('skin_cancer_model.pkl')
        scaler = joblib.load('skin_cancer_scaler.pkl')
        encoders = joblib.load('skin_cancer_encoders.pkl')
        return model, scaler, encoders
    except Exception as e:
        return None, None, str(e)

@st.cache_resource
def load_lung_assets():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        le_gender = joblib.load('le_gender.pkl')
        le_target = joblib.load('le_target.pkl')
        return model, scaler, le_gender, le_target
    except Exception as e:
        return None, None, None, str(e)

@st.cache_resource
def load_breast_assets():
    try:
        model = joblib.load('breast_cancer_model.pkl')
        scaler = joblib.load('breast_cancer_scaler.pkl')
        le = joblib.load('breast_cancer_encoder.pkl')
        return model, scaler, le
    except Exception as e:
        return None, None, str(e)

# Unified Navigation using Sidebar
with st.sidebar:
    st.markdown("## 🩺 Navigation")
    page = st.radio("Select Detection System:", 
                    ["🔬 Skin Cancer", "🫁 Lung Cancer", "🎀 Breast Cancer"],
                    key="nav_radio")
    
    st.markdown("---")
    st.markdown('<div class="info-box"><h3>🏥 About</h3>'
                'This system uses advanced machine learning models to help detect early signs of cancer '
                'based on risk factors and medical data.</div>', 
                unsafe_allow_html=True)

# --- PAGE 1: SKIN CANCER ---
if "Skin Cancer" in page:
    st.markdown('<div class="header-container"><h1>🔬 Skin Cancer Risk Assessment</h1></div>', unsafe_allow_html=True)
    skin_model, skin_scaler, skin_encoders = load_skin_assets()
    
    if skin_model and not isinstance(skin_encoders, str):
        with st.form("skin_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("👤 Demographics")
                sex = st.selectbox("Sex", options=list(skin_encoders['Sex'].classes_))
                age_cat = st.selectbox("Age Category", options=list(skin_encoders['AgeCategory'].classes_))
                race = st.selectbox("Race", options=list(skin_encoders['Race'].classes_))
                bmi = st.number_input("BMI", min_value=1.0, value=25.0, key="skin_bmi")
            
            with col2:
                st.subheader("🏥 Health History")
                smoking = st.radio("Smoking", ["No", "Yes"], key="skin_smoking")
                alcohol = st.radio("Alcohol drinking", ["No", "Yes"], key="skin_alcohol")
                gen_health = st.selectbox("General Health", options=list(skin_encoders['GenHealth'].classes_))
                diff_walking = st.radio("Difficulty Walking", ["No", "Yes"], key="skin_walking")

            st.subheader("👨‍⚕️ Other Indicators")
            c1, c2, c3 = st.columns(3)
            with c1:
                heart_disease = st.radio("Heart Disease", ["No", "Yes"], key="skin_heart")
                stroke = st.radio("Stroke", ["No", "Yes"], key="skin_stroke")
            with c2:
                asthma = st.radio("Asthma", ["No", "Yes"], key="skin_asthma")
                diabetic = st.selectbox("Diabetes", options=list(skin_encoders['Diabetic'].classes_))
            with c3:
                kidney_disease = st.radio("Kidney Disease", ["No", "Yes"], key="skin_kidney")
                phys_activity = st.radio("Physical Activity", ["No", "Yes"], key="skin_activity")
            
            phys_health = st.slider("Days Physical Health was Bad", 0, 30, 0, key="skin_phys")
            ment_health = st.slider("Days Mental Health was Bad", 0, 30, 0, key="skin_ment")
            sleep_time = st.number_input("Avg Sleep Hours", 1, 24, 8, key="skin_sleep")
            
            submit_skin = st.form_submit_button("Predict Skin Cancer Risk")

        if submit_skin:
            input_dict = {
                'HeartDisease': heart_disease, 'BMI': bmi, 'Smoking': smoking,
                'AlcoholDrinking': alcohol, 'Stroke': stroke, 'PhysicalHealth': float(phys_health),
                'MentalHealth': float(ment_health), 'DiffWalking': diff_walking, 'Sex': sex,
                'AgeCategory': age_cat, 'Race': race, 'Diabetic': diabetic,
                'PhysicalActivity': phys_activity, 'GenHealth': gen_health,
                'SleepTime': float(sleep_time), 'Asthma': asthma, 'KidneyDisease': kidney_disease
            }
            
            encoded_data = {col: skin_encoders[col].transform([val])[0] if col in skin_encoders else val for col, val in input_dict.items()}
            input_df = pd.DataFrame([encoded_data])
            num_cols = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
            input_df[num_cols] = skin_scaler.transform(input_df[num_cols])
            
            feature_order = ['HeartDisease', 'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 
                            'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 
                            'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 
                            'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease']
            input_df = input_df[feature_order]
            
            prediction = skin_model.predict(input_df)[0]
            prob = skin_model.predict_proba(input_df)[0][1]
            
            if prediction == 1:
                st.markdown(f'<div class="result-container high-risk"><h2>Result: HIGH RISK</h2><p>Probability: {prob:.2%}</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-container low-risk"><h2>Result: LOW RISK</h2><p>Confidence: {1-prob:.2%}</p></div>', unsafe_allow_html=True)
    else:
        st.error(f"Waiting for Skin Cancer assets... {skin_encoders if isinstance(skin_encoders, str) else ''}")


# --- PAGE 2: LUNG CANCER ---
elif "Lung Cancer" in page:
    st.markdown('<div class="header-container"><h1>🫁 Lung Cancer Prediction System</h1></div>', unsafe_allow_html=True)
    lung_model, lung_scaler, lung_le_gender, lung_le_target = load_lung_assets()
    
    if lung_model and not isinstance(lung_le_target, str):
        with st.form("lung_form"):
            col1, col2 = st.columns(2)
            with col1:
                age_l = st.number_input("Age", 1, 120, 50, key="lung_age")
                gender_l = st.selectbox("Gender", options=["Male", "Female"], key="lung_gender")
                smoking_l = st.radio("Smoking Habit", ["No", "Yes"], key="lung_smoke")
                yellow_l = st.radio("Yellow Fingers", ["No", "Yes"], key="lung_yellow")
                anxiety_l = st.radio("Anxiety", ["No", "Yes"], key="lung_anxiety")
                pressure_l = st.radio("Peer Pressure", ["No", "Yes"], key="lung_peer")
                chronic_l = st.radio("Chronic Disease", ["No", "Yes"], key="lung_chronic")

            with col2:
                fatigue_l = st.radio("Fatigue", ["No", "Yes"], key="lung_fatigue")
                allergy_l = st.radio("Allergy", ["No", "Yes"], key="lung_allergy")
                wheeze_l = st.radio("Wheezing", ["No", "Yes"], key="lung_wheeze")
                alcohol_l = st.radio("Alcohol Consuming", ["No", "Yes"], key="lung_alcohol")
                cough_l = st.radio("Coughing", ["No", "Yes"], key="lung_cough")
                breath_l = st.radio("Shortness of Breath", ["No", "Yes"], key="lung_breath")
                swallowing_l = st.radio("Swallowing Difficulty", ["No", "Yes"], key="lung_swallow")
                chest_l = st.radio("Chest Pain", ["No", "Yes"], key="lung_chest")
            
            submit_lung = st.form_submit_button("Predict Lung Cancer Result")

        if submit_lung:
            def map_v(v): return 1 if v == "Yes" else 0
            input_lung = pd.DataFrame([{
                'GENDER': 1 if gender_l == "Female" else 0,
                'AGE': age_l,
                'SMOKING': map_v(smoking_l),
                'YELLOW_FINGERS': map_v(yellow_l),
                'ANXIETY': map_v(anxiety_l),
                'PEER_PRESSURE': map_v(pressure_l),
                'CHRONIC DISEASE': map_v(chronic_l),
                'FATIGUE ': map_v(fatigue_l),
                'ALLERGY ': map_v(allergy_l),
                'WHEEZING': map_v(wheeze_l),
                'ALCOHOL CONSUMING': map_v(alcohol_l),
                'COUGHING': map_v(cough_l),
                'SHORTNESS OF BREATH': map_v(breath_l),
                'SWALLOWING DIFFICULTY': map_v(swallowing_l),
                'CHEST PAIN': map_v(chest_l)
            }])
            input_lung[['AGE']] = lung_scaler.transform(input_lung[['AGE']])
            prediction_l = lung_model.predict(input_lung)[0]
            prob_l = lung_model.predict_proba(input_lung)[0][1]
            if prediction_l == 1:
                st.markdown(f'<div class="result-container high-risk"><h2>Prediction: LUNG CANCER DETECTED</h2><p>Confidence: {prob_l:.2%}</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-container low-risk"><h2>Prediction: NO LUNG CANCER detected</h2><p>Confidence: {1-prob_l:.2%}</p></div>', unsafe_allow_html=True)
    else:
         st.error(f"Waiting for Lung Cancer assets... {lung_le_target if isinstance(lung_le_target, str) else ''}")


# --- PAGE 3: BREAST CANCER ---
elif "Breast Cancer" in page:
    st.markdown('<div class="header-container"><h1>🎀 Breast Cancer Risk Assessment</h1></div>', unsafe_allow_html=True)
    b_model, b_scaler, b_le = load_breast_assets()
    
    if b_model and not isinstance(b_le, str):
        st.write("### 🏥 Simple Multi-Health Check for Breast Lumps")
        st.info("💡 Tip: If you have a medical report, look for the 'Mean' or 'Average' values to fill this form.")
        
        with st.form("breast_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📍 Size & Shape")
                radius = st.number_input("How big is the lump? (Average Size)", value=14.0, format="%.4f", key="br_radius", help="Often listed as 'Radius Mean' in reports.")
                texture = st.number_input("How does the surface feel? (Roughness)", value=19.0, format="%.4f", key="br_texture", help="Listed as 'Texture Mean'. Higher numbers mean a rougher surface.")
                perimeter = st.number_input("Total length around the lump", value=92.0, format="%.4f", key="br_perim", help="Listed as 'Perimeter Mean'.")
                area = st.number_input("Total space the lump takes up", value=650.0, format="%.4f", key="br_area", help="Listed as 'Area Mean'.")
                smoothness = st.number_input("How smooth is the lump?", value=0.1, format="%.4f", key="br_smooth", help="Listed as 'Smoothness Mean'.")
            
            with col2:
                st.subheader("🔬 Lump Details")
                compactness = st.number_input("How firm or dense is it?", value=0.1, format="%.4f", key="br_compact", help="Listed as 'Compactness Mean'.")
                concavity = st.number_input("Are there tiny pits or hollow spots?", value=0.1, format="%.4f", key="br_concavity", help="Listed as 'Concavity Mean'.")
                concave_pts = st.number_input("How deep are those hollow spots?", value=0.05, format="%.4f", key="br_pts", help="Listed as 'Concave Points Mean'.")
                symmetry = st.number_input("How balanced is the shape?", value=0.18, format="%.4f", key="br_symm", help="Listed as 'Symmetry Mean'.")
                fractal = st.number_input("Are the edges simple or complex?", value=0.06, format="%.4f", key="br_fractal", help="Listed as 'Fractal Dimension Mean'.")
            
            submit_breast = st.form_submit_button("📢 Predict Diagnosis Now")

        if submit_breast:
            input_b = pd.DataFrame([[radius, texture, perimeter, area, smoothness, compactness, concavity, concave_pts, symmetry, fractal]], 
                                   columns=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
                                            'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'])
            scaled_b = b_scaler.transform(input_b)
            pred_b = b_model.predict(scaled_b)[0]
            prob_b = b_model.predict_proba(scaled_b)[0]
            res_label = b_le.classes_[pred_b]
            res_prob = prob_b[pred_b]
            if res_label == 'M':
                st.markdown(f'<div class="result-container high-risk"><h2>Result: Serious Risk Detected (Malignant)</h2><p>Confidence: {res_prob:.2%}</p></div>', unsafe_allow_html=True)
                st.error("⚠️ Please see a doctor immediately for further testing (like a biopsy).")
            else:
                st.markdown(f'<div class="result-container low-risk"><h2>Result: Likely Safe (Benign)</h2><p>Confidence: {res_prob:.2%}</p></div>', unsafe_allow_html=True)
                st.success("✅ The lump looks non-cancerous, but keep checking it regularly.")
    else:
        st.error(f"Waiting for Breast Cancer assets... {b_le if isinstance(b_le, str) else ''}")


# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Multi-Cancer Detection System © 2026</p>", unsafe_allow_html=True)
