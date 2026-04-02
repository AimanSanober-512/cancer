import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load Dataset
df = pd.read_csv('breast-cancer.csv')

# 2. Features and Target
# Using 'mean' columns for easier user input in the frontend
feature_cols = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'
]
target = 'diagnosis'

X = df[feature_cols]
y = df[target]

# 3. Encoding Target (M:1, B:0)
le = LabelEncoder()
y = le.fit_transform(y)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# 8. Save Assets
joblib.dump(model, 'breast_cancer_model.pkl')
joblib.dump(scaler, 'breast_cancer_scaler.pkl')
joblib.dump(le, 'breast_cancer_encoder.pkl')
print("Breast cancer assets saved successfully!")
