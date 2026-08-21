import os
import numpy as np
from flask import Flask, request, render_template_string

# Force TensorFlow to run on CPU and reduce log verbosity
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

app = Flask(__name__)

# Load the ANN model
MODEL_PATH = "ANN_model.pkl"

try:
    # Keras models serialized via pickle can be loaded using tf.keras
    import pickle
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully via Pickle.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# High-end Executive UI Design (Dark Theme HTML/CSS)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise AI Prediction Engine</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: #111827;
            --accent-blue: #2563eb;
            --accent-glow: #3b82f6;
            --text-main: #f9fafb;
            --text-sub: #9ca3af;
            --border-color: #1f2937;
            --input-bg: #1f2937;
            --success-green: #10b981;
            --alert-red: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            overflow: hidden;
        }

        .header {
            padding: 2.5rem 2.5rem 1.5rem 2.5rem;
            border-bottom: 1px solid var(--border-color);
            background: linear-gradient(180deg, rgba(37, 99, 235, 0.08) 0%, rgba(17, 24, 39, 0) 100%);
        }

        .header h1 {
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .header h1 span {
            color: var(--accent-glow);
        }

        .header p {
            color: var(--text-sub);
            font-size: 0.95rem;
            margin-top: 6px;
        }

        .form-container {
            padding: 2.5rem;
        }

        .grid-inputs {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .input-group label {
            font-size: 0.825rem;
            font-weight: 600;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .input-group input {
            background-color: var(--input-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px 14px;
            color: #ffffff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s ease;
        }

        .input-group input:focus {
            border-color: var(--accent-glow);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
        }

        .btn-submit {
            width: 100%;
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: #ffffff;
            font-weight: 600;
            font-size: 1rem;
            padding: 14px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        }

        .btn-submit:hover {
            opacity: 0.95;
            transform: translateY(-1px);
        }

        .result-card {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 12px;
            background: #1f2937;
            border-left: 4px solid var(--accent-blue);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .result-title {
            font-size: 0.875rem;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .result-score {
            font-size: 1.8rem;
            font-weight: 700;
            color: #ffffff;
        }

        .badge {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .badge-positive {
            background: rgba(16, 185, 129, 0.15);
            color: var(--success-green);
            border: 1px solid var(--success-green);
        }

        .badge-negative {
            background: rgba(239, 68, 68, 0.15);
            color: var(--alert-red);
            border: 1px solid var(--alert-red);
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1><span>ANN</span> Intelligence Engine</h1>
        <p>Enterprise Predictive Analysis Dashboard (10-Feature Neural Model)</p>
    </div>

    <div class="form-container">
        <form method="POST" action="/predict">
            <div class="grid-inputs">
                {% for i in range(1, 11) %}
                <div class="input-group">
                    <label for="feature_{{ i }}">Feature {{ i }}</label>
                    <input type="number" step="any" id="feature_{{ i }}" name="feature_{{ i }}" 
                           value="{{ inputs[i-1] if inputs else '0.0' }}" required>
                </div>
                {% endfor %}
            </div>

            <button type="submit" class="btn-submit">Run Predictive Analysis</button>
        </form>

        {% if probability is not none %}
        <div class="result-card">
            <div>
                <div class="result-title">Model Output Probability</div>
                <div class="result-score">{{ "%.2f"|format(probability * 100) }}%</div>
            </div>
            <div>
                {% if class_label == 1 %}
                <span class="badge badge-positive">HIGH PROBABILITY (Class 1)</span>
                {% else %}
                <span class="badge badge-negative">LOW PROBABILITY (Class 0)</span>
                {% endif %}
            </div>
        </div>
        {% endif %}
    </div>
</div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, probability=None, inputs=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Model not loaded correctly.", 500

    try:
        # Extract 10 numerical inputs from form request
        input_features = [float(request.form[f"feature_{i}"]) for i in range(1, 11)]
        features_array = np.array([input_features], dtype=np.float32)

        # Predict probability using the loaded ANN model
        prediction = model.predict(features_array)
        prob = float(prediction[0][0])
        class_output = 1 if prob >= 0.5 else 0

        return render_template_string(
            HTML_TEMPLATE, 
            probability=prob, 
            class_label=class_output, 
            inputs=input_features
        )
    except Exception as e:
        return f"Error processing prediction: {str(e)}", 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
