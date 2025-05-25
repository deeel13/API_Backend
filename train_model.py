# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import joblib
import os
print(os.getcwd())

# ✅ Make sure folder exists to save model
os.makedirs("trained_data", exist_ok=True)

# ✅ Load dataset (full path to avoid file not found errors)
df = pd.read_csv('csv/Students_Grading_Dataset.csv')

# ✅ Preview column names for debugging
print("Loaded columns:", df.columns.tolist())

# ✅ Check that target column exists
label_col = 'Grade'  # Change this to your actual column name for pass/fail
if label_col not in df.columns:
    raise ValueError(f"'{label_col}' column not found in dataset. Please check the column name.")

# ✅ Encode the target column (Pass = 1, Fail = 0)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[label_col])
joblib.dump(label_encoder, 'trained_data/student_result_encoder.pkl')

# ✅ Drop the target column to isolate features
X = df.drop(columns=[label_col])

# ✅ Encode categorical features (e.g., Gender, Grades, etc.)
X = pd.get_dummies(X)

# ✅ Save feature columns for future prediction alignment
joblib.dump(X.columns.tolist(), 'trained_data/model_features.pkl')

# ✅ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Define classification model pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', MLPClassifier(
        hidden_layer_sizes=(64,),
        activation='relu',
        max_iter=2000,
        early_stopping=True,
        random_state=42
    ))
])

# ✅ Train model
pipeline.fit(X_train, y_train)

# ✅ Save model
joblib.dump(pipeline, 'trained_data/student_pass_fail_model.pkl')

# ✅ Done
print("✅ Training complete. Model saved to 'trained_data/student_pass_fail_model.pkl'")
