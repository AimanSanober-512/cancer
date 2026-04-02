import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_csv('Skin Cancer.csv')

# 2. Features and Target
target = 'SkinCancer'
binary_cols = ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']
cat_cols = ['AgeCategory', 'Race', 'Diabetic', 'GenHealth']
num_cols = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']

# 3. Preprocessing
print("Preprocessing data...")
encoders = {}

# Encode Binary Columns
for col in binary_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Encode Multi-class Categorical
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Map GenHealth manually? Actually LabelEncoder is fine for XGBoost, 
# but let's confirm the mapping if we want to be more precise.
# For now, Simple LabelEncoder for all categoricals is efficient.

X = df.drop(target, axis=1)
y = df[target]

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scaling Numerical Features
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# 6. Model Training
print("Training model (XGBoost)...")
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# 7. Evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Save Assets
print("Saving assets...")
joblib.dump(model, 'skin_cancer_model.pkl')
joblib.dump(scaler, 'skin_cancer_scaler.pkl')
joblib.dump(encoders, 'skin_cancer_encoders.pkl')
print("Assets saved: skin_cancer_model.pkl, skin_cancer_scaler.pkl, skin_cancer_encoders.pkl")
