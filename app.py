from flask import Flask, render_template, request, url_for
import numpy as np
from gaussian import GaussianNB
from linear_regression import LinearRegression
from text_classifier import NaiveBayesTextClassifier

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/gaussian')
def gaussian():
    return render_template('gaussian.html')

@app.route('/gaussian/calculate', methods=['POST'])
def calculate():
    try:
        # Get form data and validate
        train_x = request.form['train_x']
        train_y = request.form['train_y']
        test_x = request.form['test_x']

        # Convert and validate training data
        X = np.array([
            [float(x) for x in row.split(',')]
            for row in train_x.split(';')
        ])
        
        # Validate height and weight ranges
        heights = X[:, 0]
        weights = X[:, 1]
        if any(h < 100 or h > 250 for h in heights):
            raise ValueError("Height must be between 100cm and 250cm")
        if any(w < 30 or w > 200 for w in weights):
            raise ValueError("Weight must be between 30kg and 200kg")

        # Convert labels
        y = np.array([int(x) for x in train_y.split(',')])
        if not all(label in [0, 1] for label in y):
            raise ValueError("Labels must be either 0 (underweight) or 1 (overweight)")

        # Convert and validate test data
        test_data = [float(x) for x in test_x.split(',')]
        if len(test_data) != 2:
            raise ValueError("Test data must contain both height and weight")
        if test_data[0] < 100 or test_data[0] > 250:
            raise ValueError("Test height must be between 100cm and 250cm")
        if test_data[1] < 30 or test_data[1] > 200:
            raise ValueError("Test weight must be between 30kg and 200kg")
        
        test_X = np.array(test_data).reshape(1, -1)

        # Train and predict
        model = GaussianNB()
        model.fit(X, y)
        prediction = model.predict(test_X)[0]

        # Format result message
        status = "overweight" if prediction == 1 else "underweight"
        result = f"Classification: {status} (height: {test_data[0]}cm, weight: {test_data[1]}kg)"
        
        return render_template('gaussian.html', result=result)
    except Exception as e:
        return render_template('gaussian.html', result=f"Error: {str(e)}")

@app.route('/linear_regression')  # Changed from /linear to /linear_regression
def linear_regression():  # Changed function name
    return render_template('linear.html')

@app.route('/linear_regression/calculate', methods=['POST'])  # Updated route
def linear_calculate():
    try:
        # Get form data
        study_hours = np.array([float(x) for x in request.form['train_x'].split(',')]).reshape(-1, 1)
        grades = np.array([float(x) for x in request.form['train_y'].split(',')])
        test_hours = np.array([float(request.form['test_x'])]).reshape(1, -1)

        # Validate input
        if any(h < 0 or h > 24 for h in study_hours.flatten()):
            raise ValueError("Study hours must be between 0 and 24")
        if any(g < 0 or g > 100 for g in grades):
            raise ValueError("Grades must be between 0 and 100")

        # Train and predict
        model = LinearRegression()
        model.fit(study_hours, grades)
        prediction = model.predict(test_hours)[0]
        r2 = model.r2_score(grades, model.predict(study_hours))

        result = f"Predicted grade for {test_hours[0][0]} hours of study: {prediction:.1f}%"
        return render_template('linear.html', result=result, r2_score=f"{r2:.4f}")
    except Exception as e:
        return render_template('linear.html', result=f"Error: {str(e)}")

@app.route('/nlp')
def nlp():
    return render_template('nlp.html')

@app.route('/nlp/calculate', methods=['POST'])
def nlp_calculate():
    try:
        # Get form data
        train_texts = request.form['train_texts'].split(';')
        train_y = np.array([int(x) for x in request.form['train_y'].split(',')])
        test_text = request.form['test_text']

        # Validate input
        if len(train_texts) != len(train_y):
            raise ValueError("Number of training texts must match number of labels")
        if not all(label in [0, 1] for label in train_y):
            raise ValueError("Labels must be either 0 (negative) or 1 (positive)")

        # Train and predict
        model = NaiveBayesTextClassifier()
        model.fit(train_texts, train_y)
        prediction = model.predict(test_text)

        # Format result
        sentiment = model.get_class_name(prediction)
        result = f"Sentiment Analysis: {sentiment}\nText: '{test_text}'"
        
        return render_template('nlp.html', result=result)
    except Exception as e:
        return render_template('nlp.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
