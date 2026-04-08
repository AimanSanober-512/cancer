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
    print("🚀 Starting DEEP CLEANING and Improved Training...")

    # --- 1. Skin Cancer ---
    if os.path.exists('Skin Cancer.csv'):
        print("\n--- Training Deeply Cleaned Skin Cancer Model ---")
        df = pd.read_csv('Skin Cancer.csv')
        target = 'SkinCancer'
        
        # 1. Deduplication
        initial_rows = df.shape[0]
        df = df.drop_duplicates()
        print(f"   Removed {initial_rows - df.shape[0]} duplicate rows.")
        
        # 2. Outlier Handling
        num_cols = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
        df = remove_outliers_iqr(df, num_cols)
        
        # 3. Age-Balanced Oversampling (Fix Age Bias)
        younger_cats = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59']
        df_young_yes = df[(df[target] == 'Yes') & (df['AgeCategory'].isin(younger_cats))]
        if not df_young_yes.empty:
            df = pd.concat([df, df_young_yes, df_young_yes, df_young_yes])
            print(f"   Amplified {len(df_young_yes)} younger skin cancer cases for better sensitivity.")

        # 4. Label Encoding
        binary_cols = ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'PhysicalActivity', 'Asthma', 'KidneyDisease']
        cat_cols = ['AgeCategory', 'Race', 'Diabetic', 'GenHealth']
        
        encoders = {}
        for col in binary_cols + cat_cols + [target]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
        X = df.drop(target, axis=1)
        y = df[target]
        
        feature_order = ['HeartDisease', 'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 
                        'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 
                        'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 
                        'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease']
        X = X[feature_order]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])
        
        # Use more conservative parameters to avoid noise
        pos_weight = (y == 0).sum() / (y == 1).sum()
        model = XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss', 
            scale_pos_weight=pos_weight,
            random_state=42,
            n_estimators=100,
            max_depth=4, # Reduced depth for better generalization
            min_child_weight=10 # Avoid picking up patterns from single outliers
        )
        model.fit(X_train, y_train)
        
        joblib.dump(model, 'skin_model.pkl')
        joblib.dump(scaler, 'skin_scaler.pkl')
        joblib.dump(encoders, 'skin_encoders.pkl')
        print(f"✅ Skin Model Saved. Accuracy: {model.score(X_test, y_test):.2%}")

    # --- 2. Lung Cancer ---
    if os.path.exists('lung_cancer.csv'):
        print("\n--- Training Deeply Cleaned Lung Cancer Model ---")
        df = pd.read_csv('lung_cancer.csv')
        df.columns = df.columns.str.strip()
        
        # 1. Deduplication
        df = df.drop_duplicates()
        
        # Data Mappings
        le_gender = LabelEncoder()
        df['GENDER'] = le_gender.fit_transform(df['GENDER'])
        df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
        
        cols_to_map = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 
                       'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 
                       'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
        for col in cols_to_map:
            if df[col].max() == 2:
                df[col] = df[col].map({1: 0, 2: 1})
        
        # 2. Oversampling to fix the 87/13 imbalance
        df_no = df[df['LUNG_CANCER'] == 0]
        df_yes = df[df['LUNG_CANCER'] == 1]
        if len(df_no) > 0:
            # Upsample the 'NO' cases to match 'YES' cases
            df_no_upsampled = df_no.sample(len(df_yes), replace=True, random_state=42)
            df = pd.concat([df_yes, df_no_upsampled]).sample(frac=1, random_state=42) # Shuffle
            print(f"   Balanced Lung data using Oversampling: {df['LUNG_CANCER'].value_counts().to_dict()}")
        
        X = df.drop('LUNG_CANCER', axis=1)
        y = df['LUNG_CANCER']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train[['AGE']] = scaler.fit_transform(X_train[['AGE']])
        X_test[['AGE']] = scaler.transform(X_test[['AGE']])
        
        model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=5)
        model.fit(X_train, y_train)
        
        joblib.dump(model, 'lung_model.pkl')
        joblib.dump(scaler, 'lung_scaler.pkl')
        joblib.dump(le_gender, 'lung_gender_encoder.pkl')
        print(f"✅ Lung Model Saved. Accuracy: {model.score(X_test, y_test):.2%}")

    # --- 3. Breast Cancer ---
    if os.path.exists('cancer_classification.csv'):
        print("\n--- Training Deeply Cleaned Breast Cancer Model ---")
        df = pd.read_csv('cancer_classification.csv')
        target = 'benign_0__mal_1'
        
        mean_features = [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
            'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension'
        ]
        
        X = df[mean_features]
        # 1=Malignant, 0=Benign for consistency
        y = 1 - df[target]
        
        X = remove_outliers_iqr(X, mean_features)
        
        # Upsample 'Malignant' to balance with 'Benign' (if needed)
        df_temp = pd.concat([X, y], axis=1)
        df_m = df_temp[df_temp[target] == 1] # Now target is y
        df_b = df_temp[df_temp[target] == 0]
        # In reality y is already 0/1 based on 1-df[target]. Let's be careful.
        # df[target] 1=Benign, 0=Malignant. y 0=Benign, 1=Malignant.
        df_benign = df_temp[y == 0]
        df_malignant = df_temp[y == 1]
        
        if len(df_benign) > len(df_malignant):
            df_m_upsampled = df_malignant.sample(len(df_benign), replace=True, random_state=42)
            df_balanced = pd.concat([df_benign, df_m_upsampled])
            X_b = df_balanced[mean_features]
            y_b = df_balanced[target] # Actually target name was changed? No, it's the last column.
            # Best to just use y_b directly from the concat.
            y_b = df_balanced.iloc[:, -1]
        else:
            X_b, y_b = X, y

        X_train, X_test, y_train, y_test = train_test_split(X_b, y_b, test_size=0.2, random_state=42, stratify=y_b)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, max_depth=4)
        model.fit(X_train, y_train)
        
        joblib.dump(model, 'breast_model.pkl')
        joblib.dump(scaler, 'breast_scaler.pkl')
        print(f"✅ Breast Model Saved. Accuracy: {model.score(X_test, y_test):.2%}")

    print("\n✨ DEEP CLEANING complete. All models are now balanced and objective!")

if __name__ == "__main__":
    train_improved_models()
