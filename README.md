# Machine Learning Classifiers Web Application

A Flask-based web application implementing three different machine learning classifiers:
- Gaussian Naive Bayes for BMI Classification
- Linear Regression for Grade Prediction
- Naive Bayes Text Classifier for Sentiment Analysis

## Features

### 1. BMI Classification (Gaussian Naive Bayes)
- Classifies individuals as underweight/overweight based on height and weight
- Uses Gaussian Naive Bayes algorithm
- Input validation for realistic height and weight ranges

### 2. Grade Prediction (Linear Regression)
- Predicts student grades based on study hours
- Implements simple linear regression
- Includes R² score calculation
- Input validation for reasonable study hours and grades

### 3. Sentiment Analysis (Naive Bayes Text Classifier)
- Analyzes text sentiment (positive/negative)
- Uses Naive Bayes with add-one smoothing
- Text preprocessing with lowercase conversion and tokenization

## Setup and Installation

```bash
# Clone the repository
git clone <repository-url>

# Install required packages
pip install flask numpy

# Run the application
python app.py
```

## Usage

1. Access the application at `http://localhost:5000`
2. Choose a classifier from the navigation menu
3. Enter training data in the specified format
4. Submit test data for prediction

### Input Formats

#### Gaussian Naive Bayes:
- Training X: height,weight pairs separated by semicolons (e.g., "170,70;165,60")
- Training Y: comma-separated labels (0 or 1)
- Test X: single height,weight pair

#### Linear Regression:
- Training X: comma-separated study hours
- Training Y: comma-separated grades
- Test X: single study hours value

#### Text Classifier:
- Training texts: semicolon-separated sentences
- Training Y: comma-separated labels (0=negative, 1=positive)
- Test text: single sentence

## Technical Implementation

- Flask for web interface
- NumPy for numerical computations
- Custom implementations of machine learning algorithms
- Input validation and error handling

## Requirements

- Python 3.x
- Flask
- NumPy

# Machine Learning Classifiers Web Application

# ...existing code...

## How to Run

1. **Set up Python Environment**
   ```bash
   # Create a virtual environment (optional but recommended)
   python -m venv venv
   
   # Activate virtual environment
   # For Windows:
   venv\Scripts\activate
   # For Unix/MacOS:
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install flask numpy
   ```

3. **Project Structure**
   Ensure your project has the following structure:
   ```
   project2/
   ├── app.py
   ├── gaussian.py
   ├── linear_regression.py
   ├── text_classifier.py
   ├── templates/
   │   ├── home.html
   │   ├── gaussian.html
   │   ├── linear.html
   │   └── nlp.html
   └── static/
       └── styles.css
   ```

4. **Run the Application**
   ```bash
   # Navigate to project directory
   cd project2
   
   # Run Flask application
   python app.py
   ```

5. **Access the Application**
   - Open your web browser
   - Navigate to `http://localhost:5000`
   - The application will be running on your local machine

6. **Troubleshooting**
   - Make sure all required files are in place
   - Check that port 5000 is not in use
   - Ensure Flask is installed correctly
   - Check console for any error messages
