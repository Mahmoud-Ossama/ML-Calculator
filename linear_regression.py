import numpy as np

class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        # Add column of ones for intercept
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Calculate coefficients using normal equation
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
        # Extract intercept and coefficients
        self.intercept = theta[0]
        self.coefficients = theta[1:]
        return self

    def predict(self, X):
        return np.dot(X, self.coefficients) + self.intercept

    def r2_score(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
