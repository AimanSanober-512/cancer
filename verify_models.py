import joblib
import pandas as pd
import numpy as np

def verify_models():
    print("🔍 Verifying Models with Sample Inputs...")

    # 1. Lung Cancer
    print("\n--- Lung Cancer Verification ---")
    try:
        model = joblib.load('lung_model.pkl')
        scaler = joblib.load('lung_scaler.pkl')
        
        # Low risk
        low_risk = pd.DataFrame([{
            'GENDER': 0, 'AGE': 25.0, 'SMOKING': 0, 'YELLOW_FINGERS': 0, 'ANXIETY': 0, 
            'PEER_PRESSURE': 0, 'CHRONIC DISEASE': 0, 'FATIGUE': 0, 'ALLERGY': 0, 
            'WHEEZING': 0, 'ALCOHOL CONSUMING': 0, 'COUGHING': 0, 'SHORTNESS OF BREATH': 0, 
            'SWALLOWING DIFFICULTY': 0, 'CHEST PAIN': 0
        }])
        low_risk[['AGE']] = scaler.transform(low_risk[['AGE']])
        
        # High risk
        high_risk = pd.DataFrame([{
            'GENDER': 1, 'AGE': 70.0, 'SMOKING': 1, 'YELLOW_FINGERS': 1, 'ANXIETY': 1, 
            'PEER_PRESSURE': 1, 'CHRONIC DISEASE': 1, 'FATIGUE': 1, 'ALLERGY': 1, 
            'WHEEZING': 1, 'ALCOHOL CONSUMING': 1, 'COUGHING': 1, 'SHORTNESS OF BREATH': 1, 
            'SWALLOWING DIFFICULTY': 1, 'CHEST PAIN': 1
        }])
        high_risk[['AGE']] = scaler.transform(high_risk[['AGE']])
        
        print(f"Low risk prob: {model.predict_proba(low_risk)[0][1]:.2%}")
        print(f"High risk prob: {model.predict_proba(high_risk)[0][1]:.2%}")
    except Exception as e:
        print(f"Error: {e}")

    # 2. Breast Cancer
    print("\n--- Breast Cancer Verification ---")
    try:
        model = joblib.load('breast_model.pkl')
        scaler = joblib.load('breast_scaler.pkl')
        
        # Low risk (Sample values for Benign)
        low_risk = pd.DataFrame([[10.0, 15.0, 60.0, 300.0, 0.08, 0.05, 0.03, 0.02, 0.15, 0.05]], 
                               columns=['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 
                                        'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension'])
        low_scaled = scaler.transform(low_risk)
        
        # High risk (Sample values for Malignant)
        high_risk = pd.DataFrame([[20.0, 25.0, 130.0, 1200.0, 0.12, 0.25, 0.3, 0.15, 0.25, 0.08]], 
                                columns=['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 
                                         'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension'])
        high_scaled = scaler.transform(high_risk)
        
        print(f"Low risk prob: {model.predict_proba(low_scaled)[0][1]:.2%}")
        print(f"High risk prob: {model.predict_proba(high_scaled)[0][1]:.2%}")
    except Exception as e:
        print(f"Error: {e}")

    # 3. Skin Cancer
    print("\n--- Skin Cancer Verification ---")
    try:
        model = joblib.load('skin_model.pkl')
        scaler = joblib.load('skin_scaler.pkl')
        encoders = joblib.load('skin_encoders.pkl')
        
        def encode(input_dict):
            encoded = {}
            for col, val in input_dict.items():
                if col in encoders: encoded[col] = encoders[col].transform([str(val)])[0]
                else: encoded[col] = val
            df = pd.DataFrame([encoded])
            df[['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']] = scaler.transform(df[['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']])
            order = ['HeartDisease', 'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease']
            return df[order]

        low_risk_data = {
            'HeartDisease': 'No', 'BMI': 22.0, 'Smoking': 'No', 'AlcoholDrinking': 'No', 'Stroke': 'No',
            'PhysicalHealth': 0.0, 'MentalHealth': 0.0, 'DiffWalking': 'No', 'Sex': 'Male',
            'AgeCategory': '25-29', 'Race': 'White', 'Diabetic': 'No', 'PhysicalActivity': 'Yes',
            'GenHealth': 'Excellent', 'SleepTime': 8.0, 'Asthma': 'No', 'KidneyDisease': 'No'
        }
        
        high_risk_data = {
            'HeartDisease': 'Yes', 'BMI': 35.0, 'Smoking': 'Yes', 'AlcoholDrinking': 'Yes', 'Stroke': 'Yes',
            'PhysicalHealth': 15.0, 'MentalHealth': 5.0, 'DiffWalking': 'Yes', 'Sex': 'Female',
            'AgeCategory': '80 or older', 'Race': 'White', 'Diabetic': 'Yes', 'PhysicalActivity': 'No',
            'GenHealth': 'Poor', 'SleepTime': 4.0, 'Asthma': 'Yes', 'KidneyDisease': 'Yes'
        }
        
        low_in = encode(low_risk_data)
        high_in = encode(high_risk_data)
        
        print(f"Low risk prob: {model.predict_proba(low_in)[0][1]:.2%}")
        print(f"High risk prob: {model.predict_proba(high_in)[0][1]:.2%}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_models()
