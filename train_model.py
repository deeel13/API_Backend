# trained_model.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

# 📁 Ensure the output folder exists
OUTPUT_DIR = "trained_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 📥 Load and preprocess dataset
def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    # Apply Pass/Fail rule
    df['PassFail'] = df.apply(
        lambda row: 'Pass' if (
            row['Attendance (%)'] >= 75 and 
            row['Participation_Score'] >= 5 and 
            row['Study_Hours_per_Week'] >= 7
        ) else 'Fail',
        axis=1
    )

    # Define features and encode target
    X = df[['Attendance (%)', 'Participation_Score', 'Study_Hours_per_Week', 'Sleep_Hours_per_Night']]
    y = df['PassFail'].map({'Fail': 0, 'Pass': 1})
    return train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Build the pipeline
def build_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', MLPClassifier(
            hidden_layer_sizes=(32,),
            activation='relu',
            max_iter=1000,
            early_stopping=True,
            random_state=42
        ))
    ])

# 🔁 Main training function
def train_and_save_model():
    X_train, X_test, y_train, y_test = load_and_preprocess_data('csv/Students_Grading_Dataset.csv')
    model_pipeline = build_pipeline()
    model_pipeline.fit(X_train, y_train)
    model_path = os.path.join(OUTPUT_DIR, 'student_pass_model.pkl')
    joblib.dump(model_pipeline, model_path)
    print(f"✅ Model training complete. Saved to '{model_path}'")

# 🚀 Run training
if __name__ == "__main__":
    train_and_save_model()
