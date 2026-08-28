def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    teta = float(x0)
    for _ in range(steps):
        gradiente = 2 * a * teta + b
        teta = teta - lr * gradiente
    return float(teta)