import os
from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer

# Ensure we're using absolute paths
base_dir = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(base_dir, 'templates')
static_dir = os.path.join(base_dir, 'static')

app = Flask(__name__, 
        template_folder=template_dir,
        static_folder=static_dir)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/gaussian', methods=['GET'])
def gaussian():
    try:
        return render_template('gaussian.html')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/linear', methods=['GET'])
def linear():
    try:
        return render_template('linear.html')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/naive_bayes', methods=['GET'])
def naive_bayes():
    try:
        return render_template('naive_bayes.html')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/solve_gaussian', methods=['POST'])
def solve_gaussian():
    try:
        X = np.array([list(map(float, x.split(','))) for x in request.form['X'].split(';')])
        y = np.array(list(map(int, request.form['y'].split(','))))
        X_pred = np.array([list(map(float, request.form['X_pred'].split(',')))])
        
        clf = GaussianNB()
        clf.fit(X, y)
        prediction = clf.predict(X_pred)
        
        return jsonify({'result': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/solve_linear', methods=['POST'])
def solve_linear():
    try:
        X = np.array([list(map(float, x.split(','))) for x in request.form['X'].split(';')])
        y = np.array(list(map(float, request.form['y'].split(','))))
        X_pred = np.array([list(map(float, request.form['X_pred'].split(',')))])
        
        reg = LinearRegression()
        reg.fit(X, y)
        prediction = reg.predict(X_pred)
        
        return jsonify({'result': float(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/solve_naive_bayes', methods=['POST'])
def solve_naive_bayes():
    try:
        training_data = request.form['training_data'].split(';')
        training_labels = request.form['training_labels'].split(',')
        test_sentence = request.form['test_sentence']
        
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(training_data)
        clf = MultinomialNB()
        clf.fit(X, training_labels)
        
        test_vector = vectorizer.transform([test_sentence])
        prediction = clf.predict(test_vector)
        
        return jsonify({'result': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Print debug information
    print("Current directory:", os.getcwd())
    print("Template directory:", template_dir)
    print("Template exists:", os.path.exists(os.path.join(template_dir, 'index.html')))
    
    app.run(debug=True)
