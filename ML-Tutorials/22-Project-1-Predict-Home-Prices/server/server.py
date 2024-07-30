#server.py
from flask import Flask, request, jsonify
import util

app = Flask(__name__)

# Load saved artifacts when the server starts
util.load_saved_artifacts()

@app.route("/get_location_names")
def get_location_names():
    response = jsonify(
        {
            'location': util.get_location_names()
        }
    )
    response.headers.add('Access-Control-Allow-Origin', '*')    
    return response

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify(
        {
            'estimated_price': util.get_estimated_price(location,total_sqft,bhk,bath)
        }
    )
    
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":
    print("Starting Flask Server for Home Prices Prediction")
    app.run()
