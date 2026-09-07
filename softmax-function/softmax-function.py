import numpy as np

def softmax(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        cambiado = x - np.max(x)
        valores_exp = np.exp(cambiado)
        return valores_exp / np.sum(valores_exp)
    cambiado = x - np.max(x, axis=1, keepdims=True)
    valores_exp = np.exp(cambiado)

    return valores_exp / np.sum(valores_exp, axis=1, keepdims=True)
