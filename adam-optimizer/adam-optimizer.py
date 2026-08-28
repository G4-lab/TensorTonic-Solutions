import numpy as np

def adam_step( param, grad, m, v, t, lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
    param = np.asarray(param, dtype=float)
    grad = np.asarray(grad, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    mt = beta1 * m + (1.0 - beta1) * grad
    vt = beta2 * v + (1.0 - beta2) * (grad * grad)
    mgorro = mt / (1.0 - beta1 ** t)
    vgorro = vt / (1.0 - beta2 ** t)
    parametros_nuevos = param - lr * (mgorro / (np.sqrt(vgorro) + eps))
    return parametros_nuevos, mt, vt