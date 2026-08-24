def polynomial_features(x_inputs, grad_power):
    resultado = [[X ** N for N in range(grad_power + 1)] for X in x_inputs]
    return resultado
    