from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the Random Forest Classifier model
filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

# Define home route
@app.route('/')
def home():
    return render_template('index.html')

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    preg = int(request.form['pregnancies'])
    glucose = int(request.form['glucose'])
    bp = int(request.form['bloodpressure'])
    st = int(request.form['skinthickness'])
    insulin = int(request.form['insulin'])
    bmi = float(request.form['bmi'])
    dpf = float(request.form['dpf'])
    age = int(request.form['age'])

    # Preprocess input data
    data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])

    # Make prediction
    prediction = classifier.predict(data)

    # Return prediction as JSON response
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
