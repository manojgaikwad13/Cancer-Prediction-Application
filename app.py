from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model only (no scaler)
model = pickle.load(open('cancer_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        bmi = float(request.form['bmi'])
        smoking = int(request.form['smoking'])
        genetic_risk = int(request.form['genetic_risk'])
        physical_activity = float(request.form['physical_activity'])
        alcohol_intake = float(request.form['alcohol_intake'])
        cancer_history = int(request.form['cancer_history'])

        # Combine inputs
        input_data = np.array([[age, gender, bmi, smoking, genetic_risk,
                                physical_activity, alcohol_intake, cancer_history]])

        # Direct prediction (no scaling)
        prediction = model.predict(input_data)[0]
        output = 'Cancer Detected' if prediction == 1 else 'No Cancer Detected'

        return render_template('result.html', result=output)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
