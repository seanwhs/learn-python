from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import tensorflow as tf

app = Flask(__name__)

# Define custom R-squared metric
def R_squared(y_true, y_pred):
    residual = tf.reduce_sum(tf.square(y_true - y_pred))
    total = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = 1.0 - residual / total
    return r2

# Define preprocessing functions
def preprocess_input(data):
    df = pd.DataFrame([data], columns=['minimum_nights', 'number_of_reviews', 'reviews_per_month',
                                       'calculated_host_listings_count', 'availability_365',
                                       'neighbourhood_group_Brooklyn', 'neighbourhood_group_Manhattan',
                                       'neighbourhood_group_Queens', 'neighbourhood_group_Staten Island',
                                       'room_type_Private room', 'room_type_Shared room'])
    df = df.fillna(0)
    return df.values.astype(np.float32)

# Load the trained model
model = tf.keras.models.load_model('models/air-bnb.h5', custom_objects={"R_squared": R_squared})

@app.route('/')
def index():
    return render_template('airbnb.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Get data posted in JSON format
        processed_data = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(processed_data)
        
        # Return prediction as JSON
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
