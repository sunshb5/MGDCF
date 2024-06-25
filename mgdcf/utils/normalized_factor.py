import numpy as np


def compute_normalized_denominator(alpha, beta, k):
    powers_of_beta = np.power(beta, np.arange(k + 1))
    return np.power(beta, k) + alpha * np.sum(powers_of_beta[:-1])
