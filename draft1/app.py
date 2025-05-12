import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import joblib # Import joblib

app = Flask(__name__)
CORS(app)

# Define the directory where models are stored
MODEL_DIR = './models/'

# --- Load the trained model, scaler, and encoders ---
model = None
scaler = None
label_encoders = None

try:
    model_path = os.path.join(MODEL_DIR, 'phone_price_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    encoders_path = os.path.join(MODEL_DIR, 'label_encoders.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    if not os.path.exists(encoders_path):
        raise FileNotFoundError(f"Label encoders file not found at {encoders_path}")

    # Use joblib.load as in your working script
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(encoders_path)

    print("Models, scaler, and encoders loaded successfully!")
    # --- Diagnostic Print ---
    print(f"Keys found in label_encoders dictionary: {label_encoders.keys()}")
    # ------------------------

except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Please ensure the 'models' directory exists and contains 'phone_price_model.pkl', 'scaler.pkl', and 'label_encoders.pkl'.")
except Exception as e:
    print(f"An unexpected error occurred while loading model files: {e}")

# Define the exact features the model was trained on, in the correct order
MODEL_FEATURES = ['device_brand', 'os', 'screen_size', '4g', '5g', 'rear_camera_mp',
                  'front_camera_mp', 'internal_memory', 'ram', 'battery', 'weight',
                  'release_year', 'days_used']

# Mapping from frontend input names to model feature names
FRONTEND_TO_MODEL_MAP = {
    'brand': 'device_brand',
    'os': 'os',
    'screen_size_cm': 'screen_size',
    '4g': '4g',
    '5g': '5g',
    'rear_camera_mp': 'rear_camera_mp',
    'front_camera_mp': 'front_camera_mp',
    'internal_memory_gb': 'internal_memory',
    'ram_gb': 'ram',
    'battery_mah': 'battery',
    'weight_g': 'weight',
    'release_year': 'release_year',
    'days_used': 'days_used'
}

# Define categorical and numerical features based on the training script
CATEGORICAL_COLS = ['device_brand', 'os', '4g', '5g']
NUMERICAL_COLS = ['screen_size', 'rear_camera_mp', 'front_camera_mp', 'internal_memory',
                  'ram', 'battery', 'weight', 'release_year', 'days_used']


# --- Define the prediction route ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None or label_encoders is None:
        return jsonify({'error': 'Model files not loaded. Please check the server logs.'}), 500

    data = request.get_json(force=True)

    # --- Preprocessing steps aligned with predict_phone_price.py ---
    try:
        # Create a dictionary to hold the features for prediction
        user_input = {}

        # Collect data from frontend and map to model feature names
        for frontend_key, model_key in FRONTEND_TO_MODEL_MAP.items():
            if frontend_key in data and data[frontend_key] is not None:
                 user_input[model_key] = data[frontend_key]
            else:
                # If a feature is missing from the frontend, return an error
                return jsonify({'error': f'Missing required input: {frontend_key}'}), 400

        # Convert user input to a pandas DataFrame
        input_data_for_df = {feature: user_input.get(feature) for feature in MODEL_FEATURES}
        input_df = pd.DataFrame([input_data_for_df])


        # Apply Label Encoding to categorical features
        for col in CATEGORICAL_COLS:
            if col in input_df.columns and col in label_encoders:
                le = label_encoders[col]
                try:
                    # Transform individual value, handling potential unseen data
                    # Ensure the value is a string before transforming and strip whitespace
                    input_df[col] = le.transform([str(input_df[col].iloc[0]).strip()])[0]
                except ValueError as e:
                    # Handle unseen categories by using the first class from training as a fallback
                    print(f"Warning: Unseen category for '{col}': '{input_df[col].iloc[0]}'. Using fallback: '{le.classes_[0]}'")
                    input_df[col] = le.transform([le.classes_[0]])[0] # Fallback to the first class
            elif col in input_df.columns and col not in label_encoders:
                 print(f"Error: Label encoder not found for feature '{col}' during prediction.")
                 return jsonify({'error': f'Internal error: Label encoder missing for {col}.'}), 500


        # Apply Scaling to numerical features
        if NUMERICAL_COLS and scaler:
             # Ensure numerical columns exist before scaling and handle potential NaNs
             numerical_data = input_df[NUMERICAL_COLS].astype(float).fillna(0.0) # Fill NaNs with 0.0
             input_df[NUMERICAL_COLS] = scaler.transform(numerical_data)


        # Ensure the DataFrame columns and order match the training data features
        X = input_df[MODEL_FEATURES]

        # Make the prediction
        predicted_normalized_price = model.predict(X)[0]

        # Reverse the log-transformation to get the actual price
        actual_price = np.exp(predicted_normalized_price)

        # Return the predicted price
        return jsonify({'predicted_price': float(actual_price)})

    except KeyError as e:
        return jsonify({'error': f'Missing expected key during processing: {e}. Ensure all required fields are submitted.'}), 400
    except ValueError as e:
         return jsonify({'error': f'Invalid input value: {e}'}), 400
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': f'An internal error occurred during prediction: {e}'}), 500

# --- Serve the HTML file ---
@app.route('/')
def index():
    if model is None or scaler is None or label_encoders is None:
         return "<h1>Error: Model files not loaded. Please check server logs.</h1>", 500
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
