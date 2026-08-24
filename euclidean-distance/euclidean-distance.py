import numpy as np

def euclidean_distance(x, y):
    X = np.asarray(x, dtype=float)
    Y = np.asarray(y, dtype=float)
    diferencia = X - Y

    return float(np.sqrt(np.sum(diferencia ** 2)))