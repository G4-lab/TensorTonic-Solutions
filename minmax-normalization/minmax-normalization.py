import numpy as np

def minmax_scale(X, axis=0, eps = 1e-12):
    X = np.asarray(X, dtype=float)
    minimo = np.min(X, axis=axis, keepdims=True)
    maximo = np.max(X, axis=axis, keepdims=True)
    rango = (maximo - minimo) + eps
    return (X - minimo) / rango
    
    
    
   