import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Define number of rows
n = 1000

# Generate synthetic features
study_hours = np.random.randint(1, 7, n)  # 1 to 6 hours
attendance = np.random.randint(20, 101, n)  # 20% to 100%
homework_completed = np.random.choice(['Yes', 'No'], n, p=[0.7, 0.3])  # 70% Yes
parental_education = np.random.choice(['High School', 'Bachelor', 'Master'], n, p=[0.4, 0.4, 0.2])

# Rule-based logic to generate realistic labels
pass_fail = []
for s, a, h in zip(study_hours, attendance, homework_completed):
    if s >= 3 and a >= 60 and h == 'Yes':
        pass_fail.append(1)
    else:
        pass_fail.append(0)

# Create DataFrame
df = pd.DataFrame({
    'study_hours': study_hours,
    'attendance_percentage': attendance,
    'homework_completed': homework_completed,
    'parental_education': parental_education,
    'pass_fail': pass_fail
})

# Save as CSV

df.to_csv("student_data.csv", index=False)
print("✅ 1000-row dataset saved as 'student_data.csv'")

# Step 2: Import necessary libraries

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
# Step 3: Load and preview the dataset

df = pd.read_csv("student_data.csv")
df.head()
# Step 4: Encode categorical variables

label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

df.head()
# Step 5: Split data into features and target

X = df.drop("pass_fail", axis=1)  # Features
y = df["pass_fail"]               # Target

# Step 6: Train-test split and scaling

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Train the ML model

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 8: Predict and evaluate

y_pred = model.predict(X_test_scaled)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("📄 Classification Report:\n", classification_report(y_test, y_pred))

# Step 9: Save the model and scaler

joblib.dump(model, "student_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler saved successfully.")