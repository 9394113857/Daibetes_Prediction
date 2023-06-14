from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the dumped model
filename = 'diabetes-prediction-rfc-model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    pregnancies = int(request.form['pregnancies'])
    glucose = int(request.form['glucose'])
    blood_pressure = int(request.form['blood_pressure'])
    skin_thickness = int(request.form['skin_thickness'])
    insulin = int(request.form['insulin'])
    bmi = float(request.form['bmi'])
    dpf = float(request.form['dpf'])
    age = int(request.form['age'])

    # Create a DataFrame from user input
    input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                            columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age'])

    # Make predictions
    prediction = loaded_model.predict(input_df)

    # Display the prediction
    if prediction[0] == 0:
        result = "The person is not diabetic."
    else:
        result = "The person is diabetic."

    return render_template('results.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
