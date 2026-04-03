import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

print("Starting training script...")

# 1. Skin Cancer
if os.path.exists('Skin Cancer.csv'):
    print("Training Skin Cancer Model...")
    df_skin = pd.read_csv('Skin Cancer.csv')
    target = 'SkinCancer'
    binary_cols = ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']
    cat_cols = ['AgeCategory', 'Race', 'Diabetic', 'GenHealth']
    num_cols = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
    skin_encoders = {}
    for col in binary_cols + cat_cols:
        le = LabelEncoder()
        df_skin[col] = le.fit_transform(df_skin[col].astype(str))
        skin_encoders[col] = le
    X = df_skin.drop(target, axis=1)
    y = df_skin[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    skin_scaler = StandardScaler()
    X_train.loc[:, num_cols] = skin_scaler.fit_transform(X_train[num_cols])
    skin_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    skin_model.fit(X_train, y_train)
    joblib.dump(skin_model, 'skin_model.pkl')
    joblib.dump(skin_scaler, 'skin_scaler.pkl')
    joblib.dump(skin_encoders, 'skin_encoders.pkl')
    print("Skin Model Saved.")

# 2. Lung Cancer
if os.path.exists('lung_cancer.csv'):
    print("Training Lung Cancer Model...")
    df_lung = pd.read_csv('lung_cancer.csv')
    df_lung.columns = df_lung.columns.str.strip()
    le_gender = LabelEncoder()
    df_lung['GENDER'] = le_gender.fit_transform(df_lung['GENDER'])
    le_target = LabelEncoder()
    df_lung['LUNG_CANCER'] = le_target.fit_transform(df_lung['LUNG_CANCER'])
    cols_to_map = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 
                   'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 
                   'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
    for col in cols_to_map:
        df_lung[col] = df_lung[col].map({1: 0, 2: 1})
    X = df_lung.drop('LUNG_CANCER', axis=1)
    y = df_lung['LUNG_CANCER']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lung_scaler = StandardScaler()
    X_train[['AGE']] = lung_scaler.fit_transform(X_train[['AGE']])
    lung_model = RandomForestClassifier(n_estimators=100, random_state=42)
    lung_model.fit(X_train, y_train)
    joblib.dump(lung_model, 'lung_model.pkl')
    joblib.dump(lung_scaler, 'lung_scaler.pkl')
    joblib.dump(le_gender, 'lung_gender_encoder.pkl')
    joblib.dump(le_target, 'lung_target_encoder.pkl')
    print("Lung Model Saved.")

# 3. Breast Cancer
if os.path.exists('cancer_classification.csv'):
    print("Training Breast Cancer Model...")
    df_breast = pd.read_csv('cancer_classification.csv')
    target = 'benign_0__mal_1'
    X = df_breast.drop(target, axis=1)
    y = df_breast[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    breast_scaler = StandardScaler()
    X_train = breast_scaler.fit_transform(X_train)
    breast_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    breast_model.fit(X_train, y_train)
    joblib.dump(breast_model, 'breast_model.pkl')
    joblib.dump(breast_scaler, 'breast_scaler.pkl')
    print("Breast Model Saved.")

print("All models trained and saved successfully!")
