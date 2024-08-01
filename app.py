from flask import Flask, request, jsonify
import numpy as np
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler2.pkl', 'rb'))


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Validate and extract data, providing default values if necessary
    try:
        temperature = float(data.get('temperature', 0))
        rainfall = float(data.get('rainfall', 0))
        soil_moisture = float(data.get('soil_moisture', 0)) if data.get('soil_moisture') else 0
        crop_type = data.get('crop_type', 'unknown').lower()
        soil_type = data.get('soil_type', 'unknown').lower()
    except ValueError:
        return jsonify({'error': 'Invalid input data'}), 400

    # Encoding
    crop_type_encoding = {'corn': 0, 'rice': 1, 'soybean': 2, 'unknown': 3}
    crop_type = crop_type_encoding.get(crop_type, 3)
    soil_type_encoding = {'clay': 0, 'loamy': 1, 'sandy': 2, 'unknown': 3}
    soil_type = soil_type_encoding.get(soil_type, 3)

    # Prepare feature array and scale
    features = np.array([[temperature, rainfall, soil_moisture, crop_type, soil_type]])
    features = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)