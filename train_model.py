import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv('lung_cancer.csv')

# Preprocessing
# 1. Encode GENDER (M:0, F:1)
le_gender = LabelEncoder()
df['GENDER'] = le_gender.fit_transform(df['GENDER'])

# 2. Encode LUNG_CANCER (NO:0, YES:1)
le_target = LabelEncoder()
df['LUNG_CANCER'] = le_target.fit_transform(df['LUNG_CANCER'])

# 3. Features and Target
X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

# 4. Standardize binary features (1/2 -> 0/1)
# Most columns are 1/2, let's convert them to 0/1
binary_cols = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
               'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 
               'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 
               'SWALLOWING DIFFICULTY', 'CHEST PAIN']

for col in binary_cols:
    X[col] = X[col] - 1

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Scaling (Age)
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[['AGE']] = scaler.fit_transform(X_train[['AGE']])
X_test_scaled[['AGE']] = scaler.transform(X_test[['AGE']])

# 7. Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 8. Evaluation
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# 9. Save Model and Preprocessing Objects
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_gender, 'le_gender.pkl')
joblib.dump(le_target, 'le_target.pkl')

print("Model and scalers saved successfully!")
