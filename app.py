from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# 🔄 Load trained model and encoder
model = joblib.load('trained_data/student_pass_fail_model.pkl')
label_encoder = joblib.load('trained_data/label_encoder.pkl')
features = joblib.load('trained_data/model_features.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received JSON:", data)

        # Convert JSON to DataFrame
        input_df = pd.DataFrame([data])
        print("Initial input DataFrame:\n", input_df)

        # Manually one-hot encode 'grade' categorical feature if present
        if 'grade' in input_df.columns:
            grade_dummies = pd.get_dummies(input_df['grade'], prefix='grade')
            input_df = input_df.drop('grade', axis=1)
            input_df = pd.concat([input_df, grade_dummies], axis=1)
        print("DataFrame after one-hot encoding 'grade':\n", input_df)

        # Reindex to match model features, fill missing columns with 0
        input_encoded = input_df.reindex(columns=features, fill_value=0)
        print("Final input encoded DataFrame:\n", input_encoded)

        # Predict Pass/Fail
        prediction_encoded = model.predict(input_encoded)[0]
        print("Encoded prediction:", prediction_encoded)

        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
        print("Decoded prediction label:", prediction_label)

        return jsonify({
            "prediction": prediction_label.lower()
        })

    except Exception as e:
        # Return error message with 400 status code
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
