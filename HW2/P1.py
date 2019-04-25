import numpy as np
from math import sqrt
from math import pi
from math import exp


def generate_samples(sample_size, J):
    samples = np.zeros(sample_size)
    n_size = int(sample_size / J)
    n1_samples = np.random.normal(0.5, sqrt(0.02), size=n_size)
    n2_samples = np.random.normal(1.2, sqrt(0.09), size=n_size)
    n3_samples = np.random.normal(2.1, sqrt(0.06), size=n_size)
    n4_samples = np.random.normal(3.4, sqrt(0.04), size=n_size)
    n5_samples = np.random.normal(4.0, sqrt(0.01), size=n_size)
    for i in range(n_size):
        samples[i * 5] = n1_samples[i]
        samples[i * 5 + 1] = n2_samples[i]
        samples[i * 5 + 2] = n3_samples[i]
        samples[i * 5 + 3] = n4_samples[i]
        samples[i * 5 + 4] = n5_samples[i]
    return samples


def f(x, mu, sigma2):
    return (1 / sqrt(2 * pi * sigma2)) * \
           (exp((-1 * (x - mu) ** 2) / (2 * sigma2)))


def update_p_j_theta(x, p, mu, sigma2, j, J):
    f_x_js = np.array([f(x, mu[i], sigma2[i]) for i in range(J)])
    next_p_j_theta = (f(x, mu[j], sigma2[j]) * p[j]) / (np.sum(p * f_x_js))
    return next_p_j_theta


def update_p_js(p_j_theta, landa):
    p_next = (np.sum(p_j_theta)) / landa
    return p_next


def update_mu_j(p_j_theta, samples):
    mu_next = np.sum(p_j_theta * samples) / np.sum(p_j_theta)
    return mu_next


def update_sigma2_j(p_j_theta, samples, mu_j):
    next_sigma2 = (np.sum(p_j_theta * ((samples - mu_j) ** 2))) \
                  / np.sum(p_j_theta)
    return next_sigma2


def expectation_maximization(J, landa, iterations):
    p = np.array([1.0 / J for _ in range(J)], dtype=np.float128)
    mu = np.array([i for i in range(J)], dtype=np.float128)
    sigma2 = np.ones(J, dtype=np.float128)
    p_j_theta = np.zeros((J, len(samples)), dtype=np.float128)
    for j in range(J):
        for q in range(len(samples)):
            p_j_theta[j][q] = update_p_j_theta(samples[q], p, mu, sigma2, j, J)
    for _ in range(iterations):
        for j in range(J):
            p[j] = update_p_js(p_j_theta[j], landa)
            mu[j] = update_mu_j(p_j_theta[j], samples)
            sigma2[j] = update_sigma2_j(p_j_theta[j], samples, mu[j])
            for q in range(len(samples)):
                p_j_theta[j][q] = update_p_j_theta(samples[q], p, mu, sigma2, j, J)
    return p, mu, sigma2


if __name__ == "__main__":
    samples = generate_samples(5000, 5)
    p, mu, sigma2 = expectation_maximization(5, 5000, 200)
    print("I = 5, samples# = 5000")
    print("mu: ", mu)
    print("sigma^2: ", sigma2)
    print("pi: ", p)
    print("=========================")
    p, mu, sigma2 = expectation_maximization(2, 5000, 200)
    print("I = 2, samples# = 5000")
    print("mu: ", mu)
    print("sigma^2: ", sigma2)
    print("pi: ", p)
    print("=========================")

    p, mu, sigma2 = expectation_maximization(4, 5000, 200)
    print("I = 4, samples# = 5000")
    print("mu: ", mu)
    print("sigma^2: ", sigma2)
    print("pi: ", p)
    print("=========================")
    samples = generate_samples(1000, 5)
    p, mu, sigma2 = expectation_maximization(5, 1000, 200)
    print("I = 5, samples# = 1000")
    print("mu: ", mu)
    print("sigma^2: ", sigma2)
    print("pi: ", p)


