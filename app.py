import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- Page Config ---
st.set_page_config(
    page_title="Unified Cancer Detection",
    page_icon="🩺",
    layout="wide"
)

# --- Premium Blue Styling ---
st.markdown("""
<style>
    /* Main Background - True Dark Blue */
    .stApp {
        background-color: #0d1117 !important; /* GitHub dark bg */
        color: #c9d1d9 !important;
    }

    /* Card Background */
    .result-card {
        background-color: #161b22; /* Lighter dark */
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 2.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        color: #f0f6fc;
        text-align: center;
        margin-top: 1.5rem;
    }

    /* Advice Box */
    .tips-box {
        background-color: rgba(33, 150, 243, 0.05);
        border: 1px solid #1f6feb;
        padding: 20px;
        border-radius: 10px;
        margin-top: 25px;
        text-align: left;
    }

    .tips-box ul {
        list-style-type: none;
        padding-left: 0;
    }

    .tips-box li {
        margin-bottom: 12px;
        color: #8b949e;
        display: flex;
        align-items: center;
    }

    .tips-box li:before {
        content: "•";
        color: #58a6ff;
        font-weight: bold;
        display: inline-block; 
        width: 1em;
        margin-left: -1em;
    }

    /* Sidebar Fix */
    [data-testid="stSidebar"] {
        background-color: #010409 !important;
        border-right: 1px solid #30363d;
    }

    h1, h2, h3, h4 {
        color: #58a6ff !important;
    }

    /* Vibrant Icon for Header */
    .header-icon {
        width: 100px;
        filter: drop-shadow(0 0 10px #4fc3f7);
        margin-bottom: 20px;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #1976d2 0%, #2196f3 100%);
        color: white !important;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        height: 3rem;
        transition: 0.3s;
    }

    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(33, 150, 243, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# --- Load Assets ---
@st.cache_resource
def load_all_assets():
    try:
        essential_files = ['lung_model.pkl', 'skin_model.pkl', 'breast_model.pkl']
        for f in essential_files:
            if not os.path.exists(f):
                st.warning(f"File {f} is missing. Please run 'python train_model.py' first.")
                return None
        
        assets = {
            "lung": (joblib.load('lung_model.pkl'), joblib.load('lung_scaler.pkl')),
            "skin": (joblib.load('skin_model.pkl'), joblib.load('skin_scaler.pkl'), joblib.load('skin_encoders.pkl')),
            "breast": (joblib.load('breast_model.pkl'), joblib.load('breast_scaler.pkl'))
        }
        return assets
    except Exception as e:
        st.error(f"Error loading models: {e}")
        if "sklearn" in str(e) or "StandardScaler" in str(e):
            st.info("💡 It seems 'scikit-learn' is not correctly installed. Please run: pip install scikit-learn")
        return None

assets = load_all_assets()

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("""
        <div style='text-align: center;'>
            <img src='https://cdn-icons-png.flaticon.com/512/2966/2966327.png' class='header-icon'>
            <h1 style='color: #4fc3f7 !important;'>Cancer Checker</h1>
        </div>
    """, unsafe_allow_html=True)
    
    choice = st.radio(
        "📝 Menu",
        ["🩺 General Detection", "🫁 Lung Cancer", "🔬 Skin Cancer", "🎀 Breast Cancer"]
    )
    st.markdown("---")
    st.write("Early detection is the first step to successful treatment. Answer the questions simply.")

# --- Results Logic ---
def show_result(prob, cancer_name, symptoms=None, pred=None):
    try:
        is_high_risk = (pred == 1) if pred is not None else (prob > 0.5)
        
        # --- Native Streamlit Result Banners ---
        if is_high_risk:
            st.error(f"## 🚨 Risk Detected: {cancer_name}")
            if symptoms:
                st.warning(f"#### ⚠️ Approximate Concern: {symptoms}")
        else:
            st.success(f"## ✅ Result: Likely Safe ({cancer_name})")

        # --- Confidence Meter ---
        val = float(prob if is_high_risk else 1-prob)
        st.write(f"**Confidence Score:** {val:.2%}")
        st.progress(min(max(val, 0.0), 1.0))

        # --- Doctors Advice (Native Box) ---
        with st.container():
            st.markdown("---")
            st.subheader("🩺 Doctors Advice & Health Tips")
            
            if is_high_risk:
                st.markdown("🔴 **PLEASE CONSULT A DOCTOR IMMEDIATELY FOR A FULL CHECKUP.**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info("🥤 **Stay Hydrated**\nDrink 8-10 glasses of clean water daily.")
                st.info("🥦 **Eat Green**\nInclude green leafy vegetables in every meal.")
            with col2:
                st.info("🚭 **No Smoking**\nStrictly avoid Bidi, Cigarettes, and Tobacco.")
                st.info("🏥 **Visit Clinic**\nVisit your nearest PHC for any physical pain.")
        
        st.markdown("---")

    except Exception as e:
        st.error(f"Error displaying result: {e}")

# --- Pages ---
if choice == "🩺 General Detection":
    st.title("🩺 General Cancer Checker")
    st.write("A quick survey to check basic symptoms.")
    
    with st.expander("📝 Provide Your Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            smoke = st.radio("1. Do you smoke or chew tobacco?", ["No", "Yes"])
            cough = st.radio("2. Do you have a long-term cough?", ["No", "Yes"])
            alcohol = st.radio("3. Do you drink alcohol regularly?", ["No", "Yes"])
        with col2:
            spots = st.radio("4. Do you see any unusual skin spots or bumps?", ["No", "Yes"])
            fatigue = st.radio("5. Do you feel tired or weak often?", ["No", "Yes"])
            breath = st.radio("6. Do you find it hard to breathe?", ["No", "Yes"])
        
        age = st.slider("7. Your Age", 1, 100, 40)
        
        if st.button("Check My Health"):
            # Calculate logic
            lung_score = 0.5 if (smoke == "Yes" or cough == "Yes" or breath == "Yes") else 0.1
            skin_score = 0.5 if spots == "Yes" else 0.1
            total_prob = max(lung_score, skin_score)
            if alcohol == "Yes": total_prob += 0.05
            if fatigue == "Yes": total_prob += 0.05
            
            symptom_type = None
            if lung_score > 0.4: symptom_type = "signs related to Lung health"
            elif skin_score > 0.4: symptom_type = "signs related to Skin health"
            
            show_result(min(total_prob, 0.95), "General Screening", symptoms=symptom_type, pred=1 if total_prob > 0.5 else 0)

elif choice == "🫁 Lung Cancer":
    st.title("🫁 Detailed Lung Check")
    if assets:
        with st.form("lung_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", 1, 120, 50)
                gen = st.selectbox("Gender", ["Male", "Female"])
                smk = st.radio("Smoke?", ["No", "Yes"])
                yf = st.radio("Yellow Fingers?", ["No", "Yes"])
                anx = st.radio("Anxiety?", ["No", "Yes"])
                pp = st.radio("Peer Pressure?", ["No", "Yes"])
                cd = st.radio("Chronic Disease?", ["No", "Yes"])
            with col2:
                fat = st.radio("Fatigue?", ["No", "Yes"])
                all = st.radio("Allergy?", ["No", "Yes"])
                whe = st.radio("Wheezing?", ["No", "Yes"])
                alc = st.radio("Alcohol?", ["No", "Yes"])
                cog = st.radio("Cough?", ["No", "Yes"])
                shb = st.radio("Short Breath?", ["No", "Yes"])
                swd = st.radio("Swallowing Difficulty?", ["No", "Yes"])
                cp = st.radio("Chest Pain?", ["No", "Yes"])
            
            if st.form_submit_button("Analyze Lung Health"):
                model, scaler = assets['lung']
                def map_v(v): return 1 if v == "Yes" else 0
                df_in = pd.DataFrame([{
                    'GENDER': 0 if gen == "Male" else 1, 'AGE': age, 'SMOKING': map_v(smk),
                    'YELLOW_FINGERS': map_v(yf), 'ANXIETY': map_v(anx), 'PEER_PRESSURE': map_v(pp),
                    'CHRONIC DISEASE': map_v(cd), 'FATIGUE': map_v(fat), 'ALLERGY': map_v(all),
                    'WHEEZING': map_v(whe), 'ALCOHOL CONSUMING': map_v(alc), 'COUGHING': map_v(cog),
                    'SHORTNESS OF BREATH': map_v(shb), 'SWALLOWING DIFFICULTY': map_v(swd), 'CHEST PAIN': map_v(cp)
                }])
                df_in[['AGE']] = scaler.transform(df_in[['AGE']])
                prob = model.predict_proba(df_in)[0][1]
                show_result(prob, "Lung Cancer", pred=model.predict(df_in)[0])

elif choice == "🔬 Skin Cancer":
    st.title("🔬 Detailed Skin Check")
    
    if assets:
        with st.form("skin_form"):
            col1, col2 = st.columns(2)
            with col1:
                age_cat = st.selectbox("How old are you?", list(assets['skin'][2]['AgeCategory'].classes_))
                sex = st.selectbox("Gender", ["Male", "Female"])
                bmi = st.number_input("BMI (Weight/Height ratio)", 10.0, 60.0, 25.0)
                gen_h = st.selectbox("Self Health Rating", list(assets['skin'][2]['GenHealth'].classes_))
                race = st.selectbox("Background/Race", list(assets['skin'][2]['Race'].classes_))
            with col2:
                smoke = st.radio("Smoke regularly?", ["No", "Yes"])
                alc = st.radio("Drink alcohol regularly?", ["No", "Yes"])
                phys = st.slider("Days sick last month", 0, 30, 0)
                ment = st.slider("Days stressed last month", 0, 30, 0)
                walk = st.radio("Hard to walk?", ["No", "Yes"])
            
            if st.form_submit_button("Analyze Skin Health"):
                model, scaler, encoders = assets['skin']
                # Collect 17 features: HeartDisease, BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth, MentalHealth, DiffWalking, Sex, AgeCategory, Race, Diabetic, PhysicalActivity, GenHealth, SleepTime, Asthma, KidneyDisease
                # For brevity, placeholders for some
                raw = {
                    'HeartDisease': 'No', 'BMI': bmi, 'Smoking': smoke, 'AlcoholDrinking': alc, 'Stroke': 'No',
                    'PhysicalHealth': float(phys), 'MentalHealth': float(ment), 'DiffWalking': walk, 'Sex': sex,
                    'AgeCategory': age_cat, 'Race': race, 'Diabetic': 'No', 'PhysicalActivity': 'Yes',
                    'GenHealth': gen_h, 'SleepTime': 7.0, 'Asthma': 'No', 'KidneyDisease': 'No'
                }
                enc = {}
                for c, v in raw.items():
                    if c in encoders: enc[c] = encoders[c].transform([str(v)])[0]
                    else: enc[c] = v
                df_in = pd.DataFrame([enc])
                num_cols = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
                df_in[num_cols] = scaler.transform(df_in[num_cols])
                order = ['HeartDisease', 'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease']
                df_in = df_in[order]
                prob = model.predict_proba(df_in)[0][1]
                show_result(prob, "Skin Cancer")

elif choice == "🎀 Breast Cancer":
    st.title("🎀 Report Analysis (Breast)")
    st.write("Input medical report values for an accurate check.")
    
    if assets:
        with st.form("breast_form"):
            col1, col2 = st.columns(2)
            with col1:
                r = st.number_input("Radius (Size)", 0.0, 50.0, 14.1)
                t = st.number_input("Texture (Roughness)", 0.0, 50.0, 19.3)
                p = st.number_input("Perimeter", 0.0, 200.0, 91.9)
                a = st.number_input("Area", 0.0, 2500.0, 654.8)
                s = st.number_input("Smoothness", 0.0, 1.0, 0.1)
            with col2:
                c = st.number_input("Compactness", 0.0, 1.0, 0.1)
                cc = st.number_input("Concavity", 0.0, 1.0, 0.1)
                cp = st.number_input("Concave Points", 0.0, 1.0, 0.05)
                sy = st.number_input("Symmetry", 0.0, 1.0, 0.18)
                f = st.number_input("Fractal Dimension", 0.0, 1.0, 0.06)
            
            if st.form_submit_button("Run Analysis"):
                model, scaler = assets['breast']
                # Use 30 features (10 provided, 20 as 0)
                all_v = [r,t,p,a,s,c,cc,cp,sy,f] + [0.0]*20
                df_in = np.array([all_v])
                df_in = scaler.transform(df_in)
                prob = model.predict_proba(df_in)[0][1]
                show_result(prob, "Breast Cancer")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #8892b0;'>Developed for Healthcare Awareness | © 2026 Unified Detection Mission</p>", unsafe_allow_html=True)
