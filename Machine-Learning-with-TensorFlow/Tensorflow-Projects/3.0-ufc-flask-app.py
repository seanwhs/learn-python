from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('models/ufc-best.keras')

# Load and prepare the scaler
data = pd.read_csv('data/preprocessed_data.csv')
features = data.drop(columns=['Winner'])
scaler = StandardScaler().fit(features)

# Define a function to preprocess input data
def preprocess_input(data):
    df = pd.DataFrame(data)
    
    # Ensure all required columns are present
    missing_columns = [col for col in features.columns if col not in df.columns]
    if missing_columns:
        # Add missing columns with default values (0 or NaN)
        for col in missing_columns:
            df[col] = 0
        
    # Ensure that the DataFrame has all the columns in the same order as training data
    df = df[features.columns]
    scaled_data = scaler.transform(df)
    
    return scaled_data


@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    input_data = request.get_json()

    # Convert JSON data to DataFrame
    data = pd.DataFrame(input_data)
    
    # Preprocess data
    processed_data = preprocess_input(data)
    
    # Make predictions
    predictions = model.predict(processed_data)
    predicted_classes = (predictions > 0.5).astype(int).tolist()

    # Return the predictions as JSON
    return jsonify(predicted_classes)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'Model is up and running'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
