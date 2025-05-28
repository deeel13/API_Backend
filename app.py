from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# 🔄 Load trained model
model = joblib.load('trained_data/student_pass_model.pkl')

# ✅ Define expected features (must match training)
FEATURES = ['Attendance (%)', 'Participation_Score', 'Study_Hours_per_Week', 'Sleep_Hours_per_Night']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 📥 Load JSON input
        data = request.json
        print("Received JSON:", data)

        # ✅ Convert to DataFrame
        input_df = pd.DataFrame([data])
        print("Input DataFrame:\n", input_df)

        # ✅ Ensure all required features are present
        input_df = input_df.reindex(columns=FEATURES)
        if input_df.isnull().values.any():
            missing = input_df.columns[input_df.isnull().any()].tolist()
            return jsonify({"error": f"Missing input fields: {missing}"}), 400

        # 🧠 Predict
        prediction = model.predict(input_df)[0]
        result = "PASS" if prediction == 1 else "FAIL"
        print("Prediction:", result)

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
