import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def remove_outliers_iqr(df, columns):
    """Removes outliers using the Interquartile Range (IQR) method."""
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
    return df_clean

def train_improved_models():
    print("🚀 Starting BULLETPROOF Training (Numpy Based)...")

    # --- 1. Skin Cancer ---
    if os.path.exists('Skin Cancer.csv'):
        print("\n--- Training Skin Cancer Model ---")
        df = pd.read_csv('Skin Cancer.csv')
        target = 'SkinCancer'
        
        df = df.drop_duplicates()
        num_cols = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
        df = remove_outliers_iqr(df, num_cols)
        
        younger_cats = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59']
        df_young_yes = df[(df[target] == 'Yes') & (df['AgeCategory'].isin(younger_cats))]
        if not df_young_yes.empty:
            df = pd.concat([df, df_young_yes, df_young_yes, df_young_yes])
            print(f"   Amplified {len(df_young_yes)} younger skin cancer cases.")

        binary_cols = ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'PhysicalActivity', 'Asthma', 'KidneyDisease']
        cat_cols = ['AgeCategory', 'Race', 'Diabetic', 'GenHealth']
        
        encoders = {}
        for col in binary_cols + cat_cols + [target]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
        X = df.drop(target, axis=1)
        y = df[target]
        feature_order = ['HeartDisease', 'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease']
        X = X[feature_order]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        # Use .values to avoid storing feature names
        scaler.fit(X_train[num_cols].values)
        X_train_num_scaled = scaler.transform(X_train[num_cols].values)
        X_train.loc[:, num_cols] = X_train_num_scaled
        
        X_test_num_scaled = scaler.transform(X_test[num_cols].values)
        X_test.loc[:, num_cols] = X_test_num_scaled
        
        pos_weight = (y == 0).sum() / (y == 1).sum()
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=pos_weight, random_state=42, n_estimators=100, max_depth=4, min_child_weight=10)
        # Use .values for fit to be bulletproof
        model.fit(X_train.values, y_train.values)
        
        joblib.dump(model, 'skin_model.pkl')
        joblib.dump(scaler, 'skin_scaler.pkl')
        joblib.dump(encoders, 'skin_encoders.pkl')
        print(f"✅ Skin Model Saved. Accuracy: {model.score(X_test.values, y_test.values):.2%}")

    # --- 2. Lung Cancer ---
    if os.path.exists('lung_cancer.csv'):
        print("\n--- Training Lung Cancer Model ---")
        df = pd.read_csv('lung_cancer.csv')
        df.columns = df.columns.str.strip()
        df = df.drop_duplicates()
        
        le_gender = LabelEncoder()
        df['GENDER'] = le_gender.fit_transform(df['GENDER'])
        df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
        
        cols_to_map = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
        for col in cols_to_map:
            if df[col].max() == 2: df[col] = df[col].map({1: 0, 2: 1})
        
        df_no = df[df['LUNG_CANCER'] == 0]
        df_yes = df[df['LUNG_CANCER'] == 1]
        if len(df_no) > 0:
            df_no_upsampled = df_no.sample(len(df_yes), replace=True, random_state=42)
            df = pd.concat([df_yes, df_no_upsampled]).sample(frac=1, random_state=42)
        
        X = df.drop('LUNG_CANCER', axis=1)
        y = df['LUNG_CANCER']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        scaler.fit(X_train[['AGE']].values)
        X_train.loc[:, ['AGE']] = scaler.transform(X_train[['AGE']].values)
        X_test.loc[:, ['AGE']] = scaler.transform(X_test[['AGE']].values)
        
        model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=5)
        model.fit(X_train.values, y_train.values)
        
        joblib.dump(model, 'lung_model.pkl')
        joblib.dump(scaler, 'lung_scaler.pkl')
        joblib.dump(le_gender, 'lung_gender_encoder.pkl')
        print(f"✅ Lung Model Saved. Accuracy: {model.score(X_test.values, y_test.values):.2%}")

    # --- 3. Breast Cancer ---
    if os.path.exists('cancer_classification.csv'):
        print("\n--- Training Breast Cancer Model ---")
        df = pd.read_csv('cancer_classification.csv')
        target = 'benign_0__mal_1'
        mean_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension']
        
        X = df[mean_features]
        y = 1 - df[target] # Malignant=1
        X = remove_outliers_iqr(X, mean_features)
        
        df_temp = pd.concat([X, y], axis=1)
        df_benign = df_temp[y == 0]
        df_malignant = df_temp[y == 1]
        if len(df_benign) > len(df_malignant):
            df_m_upsampled = df_malignant.sample(len(df_benign), replace=True, random_state=42)
            df_balanced = pd.concat([df_benign, df_m_upsampled])
            X_b = df_balanced[mean_features]
            y_b = df_balanced.iloc[:, -1]
        else:
            X_b, y_b = X, y

        X_train, X_test, y_train, y_test = train_test_split(X_b, y_b, test_size=0.2, random_state=42, stratify=y_b)
        
        scaler = StandardScaler()
        # Convert to numpy before fit to avoid feature names
        scaler.fit(X_train.values)
        X_train_scaled = scaler.transform(X_train.values)
        X_test_scaled = scaler.transform(X_test.values)
        
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, max_depth=4)
        model.fit(X_train_scaled, y_train.values)
        
        joblib.dump(model, 'breast_model.pkl')
        joblib.dump(scaler, 'breast_scaler.pkl')
        print(f"✅ Breast Model Saved. Accuracy: {model.score(X_test_scaled, y_test.values):.2%}")

    print("\n✨ BULLETPROOF Training Complete. All models are now portable!")

if __name__ == "__main__":
    train_improved_models()
