from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from functools import wraps
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
try:
    model = pickle.load(open('cancer_model.pkl', 'rb'))
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

def validate_input(data):
    """Validate input data against defined constraints."""
    constraints = {
        'age': {'min': 20, 'max': 80, 'type': int},
        'gender': {'values': [0, 1], 'type': int},
        'bmi': {'min': 15, 'max': 40, 'type': float},
        'smoking': {'values': [0, 1], 'type': int},
        'genetic_risk': {'values': [0, 1, 2], 'type': int},
        'physical_activity': {'min': 0, 'max': 10, 'type': float},
        'alcohol_intake': {'min': 0, 'max': 5, 'type': float},
        'cancer_history': {'values': [0, 1], 'type': int}
    }
    
    errors = []
    for field, rules in constraints.items():
        value = data.get(field)
        
        try:
            # Type conversion
            value = rules['type'](value)
            
            # Value validation
            if 'values' in rules and value not in rules['values']:
                errors.append(f"{field} must be one of {rules['values']}")
            elif 'min' in rules and 'max' in rules:
                if value < rules['min'] or value > rules['max']:
                    errors.append(f"{field} must be between {rules['min']} and {rules['max']}")
                    
        except (ValueError, TypeError):
            errors.append(f"{field} must be a valid {rules['type'].__name__}")
    
    return errors

def handle_errors(f):
    """Decorator to handle errors in routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            return render_template('result.html', 
                                result="Error",
                                error_message="An unexpected error occurred. Please try again.")
    return decorated_function

@app.route('/')
@handle_errors
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@handle_errors
def predict():
    if model is None:
        return render_template('result.html', 
                             result="Error",
                             error_message="Model not available. Please try again later.")

    # Extract form data
    form_data = {
        'age': request.form.get('age'),
        'gender': request.form.get('gender'),
        'bmi': request.form.get('bmi'),
        'smoking': request.form.get('smoking'),
        'genetic_risk': request.form.get('genetic_risk'),
        'physical_activity': request.form.get('physical_activity'),
        'alcohol_intake': request.form.get('alcohol_intake'),
        'cancer_history': request.form.get('cancer_history')
    }

    # Validate inputs
    validation_errors = validate_input(form_data)
    if validation_errors:
        return render_template('result.html', 
                             result="Error",
                             error_message="Invalid input data: " + "; ".join(validation_errors))

    try:
        # Convert to appropriate types and create input array
        input_data = np.array([[
            int(form_data['age']),
            int(form_data['gender']),
            float(form_data['bmi']),
            int(form_data['smoking']),
            int(form_data['genetic_risk']),
            float(form_data['physical_activity']),
            float(form_data['alcohol_intake']),
            int(form_data['cancer_history'])
        ]])

        # Make prediction
        prediction = model.predict(input_data)[0]
        output = 'Cancer Detected' if prediction == 1 else 'No Cancer Detected'
        
        # Log prediction (but not personal data)
        logger.info(f"Prediction made successfully: {output}")
        
        return render_template('result.html', result=output)

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return render_template('result.html', 
                             result="Error",
                             error_message="Error processing your request. Please try again.")

@app.errorhandler(404)
def not_found_error(error):
    return render_template('result.html', 
                         result="Error",
                         error_message="Page not found."), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('result.html', 
                         result="Error",
                         error_message="Internal server error. Please try again later."), 500

if __name__ == '__main__':
    app.run(debug=True)