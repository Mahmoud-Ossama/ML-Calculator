from flask import Flask, render_template, request, url_for
import numpy as np
from gaussian import GaussianNB

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        # Get form data
        train_x = request.form['train_x']
        train_y = request.form['train_y']
        test_x = request.form['test_x']

        # Convert training data X
        X = np.array([
            [float(x) for x in row.split(',')]
            for row in train_x.split(';')
        ])

        # Convert training labels y
        y = np.array([int(x) for x in train_y.split(',')])

        # Convert test data
        test_X = np.array([float(x) for x in test_x.split(',')]).reshape(1, -1)

        # Train and predict
        model = GaussianNB()
        model.fit(X, y)
        prediction = model.predict(test_X)

        result = f"Predicted class: {prediction[0]}"
        return render_template('index.html', result=result)
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
