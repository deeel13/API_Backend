from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Flask backend is running!"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").lower()

    if user_input == "hi":
        reply = "hello"
    else:
        reply = "I didn't understand that."

    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(debug=True)
