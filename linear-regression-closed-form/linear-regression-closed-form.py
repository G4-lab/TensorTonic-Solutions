import numpy as np

def linear_regression_closed_form(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w.tolist()