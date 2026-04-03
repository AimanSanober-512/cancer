import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_all():
    print("🚀 Starting Unified Training...")

    # --- 1. Skin Cancer ---
    if os.path.exists('Skin Cancer.csv'):
        print("Training Skin Cancer Model...")
        df = pd.read_csv('Skin Cancer.csv')
        target = 'SkinCancer'
        binary_cols = ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']
        cat_cols = ['AgeCategory', 'Race', 'Diabetic', 'GenHealth']
        num_cols = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
        encoders = {}
        for col in binary_cols + cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        X = df.drop(target, axis=1)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, 'skin_model.pkl')
        joblib.dump(scaler, 'skin_scaler.pkl')
        joblib.dump(encoders, 'skin_encoders.pkl')
        print("✅ Skin Model Saved.")

    # --- 2. Lung Cancer ---
    if os.path.exists('lung_cancer.csv'):
        print("Training Lung Cancer Model...")
        df = pd.read_csv('lung_cancer.csv')
        df.columns = df.columns.str.strip()
        le_gender = LabelEncoder()
        df['GENDER'] = le_gender.fit_transform(df['GENDER'])
        le_target = LabelEncoder()
        df['LUNG_CANCER'] = le_target.fit_transform(df['LUNG_CANCER'])
        cols_to_map = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 
                       'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 
                       'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
        for col in cols_to_map:
            df[col] = df[col].map({1: 0, 2: 1})
        X = df.drop('LUNG_CANCER', axis=1)
        y = df['LUNG_CANCER']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train[['AGE']] = scaler.fit_transform(X_train[['AGE']])
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, 'lung_model.pkl')
        joblib.dump(scaler, 'lung_scaler.pkl')
        joblib.dump(le_gender, 'lung_gender_encoder.pkl')
        joblib.dump(le_target, 'lung_target_encoder.pkl')
        print("✅ Lung Model Saved.")

    # --- 3. Breast Cancer ---
    if os.path.exists('cancer_classification.csv'):
        print("Training Breast Cancer Model...")
        df = pd.read_csv('cancer_classification.csv')
        target = 'benign_0__mal_1'
        X = df.drop(target, axis=1)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, 'breast_model.pkl')
        joblib.dump(scaler, 'breast_scaler.pkl')
        print("✅ Breast Model Saved.")

    print("✨ All models trained successfully!")

if __name__ == "__main__":
    train_all()
