import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify

# Initialize Flask App
app = Flask(__name__)

# Load the trained model and feature list
print("Loading model...")
with open('sales_forecast_model.pkl', 'rb') as f:
    package = pickle.load(f)
    model = package["model"]
    feature_names = package["features"]

print(f"Model loaded. Expecting features: {feature_names}")

@app.route('/')
def home():
    return "Sales Forecasting API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get JSON data from the request
        data = request.get_json()
        
        # 2. Convert JSON to DataFrame (to match training column names)
        # We use a DataFrame so XGBoost recognizes the feature names automatically
        input_data = pd.DataFrame([data])
        
        # 3. Ensure columns match exactly
        # If the input is missing a column (like Holiday_Flag), fill it with 0
        for col in feature_names:
            if col not in input_data.columns:
                input_data[col] = 0
                
        # Reorder columns to match training order exactly
        input_data = input_data[feature_names]
        
        # 4. Make Prediction
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            'status': 'success',
            'store_id': int(data.get('Store', 0)),
            'predicted_sales': float(prediction)
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)