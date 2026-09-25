import numpy as np

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    N, D = X.shape
    w = np.zeros(D)

    b = 0.0

    for l in range(steps):
        logits = X @ w + b
        predict = _sigmoid(logits)
        grad_w = X.T @ (predict - y) / N
        grad_b = np.mean(predict - y)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b
        
    