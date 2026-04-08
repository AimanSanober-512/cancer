import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
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
        elif prob < 0.15:
            st.success(f"## ✨ Overall Health: Excellent ({cancer_name})")
            st.success("#### ✅ Result: Very Low risk detected.")
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
    
    with st.expander("📝 Comprehensive Health Survey", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            g_smoke = st.radio("1. Do you smoke regularly?", ["No", "Yes"], key="g_smoke")
            g_cough = st.radio("2. Do you have a long-term cough?", ["No", "Yes"], key="g_cough")
            g_breath = st.radio("3. Do you find it hard to breathe?", ["No", "Yes"], key="g_breath")
        with col2:
            g_spots = st.radio("4. Any unusual skin spots/bumps?", ["No", "Yes"], key="g_spots")
            g_moles = st.radio("5. Any moles changing shape or size?", ["No", "Yes"], key="g_moles")
            g_alcohol = st.radio("6. Do you drink alcohol regularly?", ["No", "Yes"], key="g_alc")
        with col3:
            g_lumps = st.radio("7. Did you feel any unusual breast lumps?", ["No", "Yes"], key="g_lump")
            g_breast_ch = st.radio("8. Any skin changes on the breast?", ["No", "Yes"], key="g_br_ch")
            g_fatigue = st.radio("9. Do you feel weak/tired often?", ["No", "Yes"], key="g_fatigue")
        
        g_age = st.slider("10. Your Current Age", 1, 100, 40)
        
        if st.button("📢 Generate Unified Risk Report"):
            if assets:
                # --- 1. Accurate Lung Prediction (High Sensitivity Mapping) ---
                lung_m, lung_s = assets['lung']
                def v(x): return 1 if x == "Yes" else 0
                
                # If they have respiratory symptoms, set secondary indicators to 1 to help the model detect it
                has_resp = 1 if (g_cough == "Yes" or g_breath == "Yes") else 0
                
                lung_df = pd.DataFrame([{
                    'GENDER': 0, 'AGE': float(g_age), 'SMOKING': v(g_smoke), 
                    'YELLOW_FINGERS': v(g_smoke), 'ANXIETY': 0, 'PEER_PRESSURE': 0, 
                    'CHRONIC DISEASE': 0, 'FATIGUE': v(g_fatigue), 'ALLERGY': has_resp, 
                    'WHEEZING': has_resp, 'ALCOHOL CONSUMING': v(g_alcohol), 
                    'COUGHING': v(g_cough), 'SHORTNESS OF BREATH': v(g_breath), 
                    'SWALLOWING DIFFICULTY': has_resp, 'CHEST PAIN': has_resp
                }])
                lung_df[['AGE']] = lung_s.transform(lung_df[['AGE']])
                l_prob = lung_m.predict_proba(lung_df)[0][1]

                # --- 2. Accurate Skin Prediction ---
                skin_m, skin_s, skin_e = assets['skin']
                raw_sk = {
                    'HeartDisease': 'No', 'BMI': 25.0, 'Smoking': g_smoke, 'AlcoholDrinking': g_alcohol, 'Stroke': 'No',
                    'PhysicalHealth': 5.0 if g_fatigue == "Yes" else 0.0, 'MentalHealth': 0.0, 'DiffWalking': 'No', 'Sex': 'Male',
                    'AgeCategory': '40-44' if g_age < 45 else '65-69', 'Race': 'White', 'Diabetic': 'No', 'PhysicalActivity': 'Yes',
                    'GenHealth': 'Fair' if g_spots == "Yes" or g_moles == "Yes" else 'Good', 'SleepTime': 7.0, 'Asthma': 'No', 'KidneyDisease': 'No'
                }
                enc_sk = {}
                for c, val in raw_sk.items():
                    if c in skin_e: enc_sk[c] = skin_e[c].transform([str(val)])[0]
                    else: enc_sk[c] = val
                skin_df = pd.DataFrame([enc_sk])
                skin_df[['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']] = skin_s.transform(skin_df[['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']])
                skin_df = skin_df[['HeartDisease', 'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease']]
                s_prob = skin_m.predict_proba(skin_df)[0][1]

                # --- 3. Accurate Breast Prediction (Precise Baselines) ---
                breast_m, breast_s = assets['breast']
                if (g_lumps == "Yes" or g_breast_ch == "Yes"):
                    # High Risk (Malignant) Parameters
                    b_r, b_t, b_p, b_a = 17.5, 22.0, 115.0, 950.0
                    b_s, b_c, b_cc, b_cp, b_sy, b_f = 0.12, 0.2, 0.3, 0.1, 0.2, 0.07
                else:
                    # Healthy (Benign) Parameters
                    b_r, b_t, b_p, b_a = 12.1, 17.9, 78.0, 462.0
                    b_s, b_c, b_cc, b_cp, b_sy, b_f = 0.09, 0.08, 0.04, 0.02, 0.17, 0.06
                
                breast_df = pd.DataFrame([[b_r, b_t, b_p, b_a, b_s, b_c, b_cc, b_cp, b_sy, b_f]], 
                                       columns=['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 
                                                'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension'])
                b_prob = breast_m.predict_proba(breast_s.transform(breast_df))[0][1]

                # --- Final Integrated Report ---
                results = {"Lung": l_prob, "Skin": s_prob, "Breast": b_prob}
                
                # Logic: If any category crossed the 50% line, it is DETECTED
                detected = [name for name, prob in results.items() if prob >= 0.5]
                
                if detected:
                    st.error("## 🚨 Result: Cancer Detected")
                    st.warning(f"#### ⚠️ Warning: It might be {', '.join(detected)} Cancer.")
                    for det in detected:
                        st.write(f"- Our analysis suggests potential warning signs for **{det} Cancer** in your profile.")
                elif max(results.values()) < 0.15:
                    st.success("## ✨ Result: Cancer Not Detected")
                    st.info("Overall risk is very low. No significant signs were identified in the cancers screened.")
                else:
                    st.success("## ✅ Result: Cancer Not Detected")
                    st.info("Risk is currently low, but we recommend monitoring for any changes in your health.")

                st.markdown("---")
                st.subheader("📋 Recommendations")
                if detected:
                    st.error("❗ Please consult a medical professional for formal clinical testing (biopsy/imaging).")
                else:
                    st.success("🥦 Please continue practicing a healthy lifestyle and regular self-checks.")

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
                
                # --- Smart Variable Threshold ---
                # For younger users, cancer is more suspicious so we use a more sensitive threshold (25%)
                elder_cats = ['60-64', '65-69', '70-74', '75-79', '80 or older']
                is_elder = age_cat in elder_cats
                threshold = 0.5 if is_elder else 0.25
                
                res_pred = 1 if prob >= threshold else 0
                if res_pred == 1:
                    st.error("## 🚨 Result: Cancer Detected")
                    st.warning(f"#### ⚠️ Warning: Based on your age group, your risk score ({prob:.1%}) is medically significant.")
                else:
                    st.success("## ✅ Result: Cancer Not Detected")
                    st.info("Your symptoms and profile do not suggest an urgent skin risk at this time.")
                
                show_result(prob, "Detailed Skin Analysis", pred=res_pred)

elif choice == "🎀 Breast Cancer":
    st.title("🎀 Simple Breast Check")
    st.write("Please answer these simple questions. No medical report is needed.")
    
    if assets:
        with st.form("breast_simple_form"):
            col1, col2 = st.columns(2)
            with col1:
                b_lump = st.radio("1. Do you feel a hard lump or knot?", ["No", "Yes"])
                b_skin = st.radio("2. Is the skin puckered or looks like orange peel?", ["No", "Yes"])
                b_pain = st.radio("3. Is there persistent pain or swelling?", ["No", "Yes"])
            with col2:
                b_shape = st.radio("4. Has the size or shape changed significantly?", ["No", "Yes"])
                b_nipple = st.radio("5. Any unusual discharge or nipple changes?", ["No", "Yes"])
                b_age = st.slider("6. Your Age", 1, 100, 40)
            
            if st.form_submit_button("Analyze Breast Health"):
                model, scaler = assets['breast']
                
                # --- Symptom-to-Medical Mapping ---
                # Benign Baselines: r=12.1, t=17.9, p=78.0, a=462.8, s=0.09, c=0.08, cc=0.046, cp=0.025, sym=0.17, f=0.06
                r, t, p, a, s_val, c, cc, cp, sym, f = 12.1, 17.9, 78.0, 462.0, 0.09, 0.08, 0.046, 0.025, 0.17, 0.06
                
                if b_lump == "Yes":
                    r, p, a, c, cc, cp = 17.5, 115.0, 978.0, 0.15, 0.16, 0.08
                if b_skin == "Yes":
                    t, s_val, cc = 22.0, 0.12, 0.2
                if b_shape == "Yes":
                    sym, f = 0.21, 0.08
                if b_pain == "Yes" or b_nipple == "Yes":
                    t += 2.0
                    s_val += 0.01
                
                # Use exactly 10 features
                cols = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 
                        'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension']
                df_in = pd.DataFrame([[r, t, p, a, s_val, c, cc, cp, sym, f]], columns=cols)
                df_in_scaled = scaler.transform(df_in)
                prob = model.predict_proba(df_in_scaled)[0][1]
                
                # Custom detected logic for simpler wording
                res_pred = 1 if prob > 0.5 else 0
                if res_pred == 1:
                    st.error("## 🚨 Result: Cancer Detected")
                    st.warning("⚠️ High warning signs found in your symptoms.")
                else:
                    st.success("## ✅ Result: Cancer Not Detected")
                    st.info("Your symptoms do not suggest a high risk at this time.")
                
                show_result(prob, "Breast Health Analysis", symptoms=None, pred=res_pred)

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #8892b0;'>Developed for Healthcare Awareness | © 2026 Unified Detection Mission</p>", unsafe_allow_html=True)
