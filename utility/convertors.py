import numpy as np


def map_x_to_psi(psi, x, should_opt_sigmas):
    if should_opt_sigmas:
        weights_size = psi.weights.size
        psi.weights = x[:weights_size]
        psi.sigmas = x[weights_size:]
    else:
        psi.weights = x


def create_x_from_psi(psi, should_opt_sigmas):
    x0 = np.copy(psi.weights)

    if should_opt_sigmas:
        x0 = np.append(x0, np.copy(psi.sigmas))

    return x0
