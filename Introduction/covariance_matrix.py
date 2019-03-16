import numpy as np


def compute_cov_matrix(samples, mean_samples):
    Q = len(samples)
    samples_ = np.array([np.array([sample]) for sample in samples])
    mean_ = np.array([mean_samples])
    cov = [np.matmul((sample - mean_).transpose(), sample - mean_) for sample in samples_]
    cov = 1/(Q-1)*sum(cov)
    return cov


if __name__ == "__main__":
    r_samples = np.array([[1.5, 0], [0, 0.5], [2, 1],
                          [1, 1], [0.5, 2], [1.5, 2],
                          [2.5, 2], [1, 3], [2, 3]])
    mean_r = np.array([1.33, 1.61])
    cov_r = compute_cov_matrix(r_samples, mean_r)

    b_samples = np.array([[-2, -1], [-1, -1], [1, -1],
                          [0.5, -0.5], [1.5, -0.5], [-1.5, 0],
                          [-0.5, 0.5], [0.5, 0.5], [1.5, 0.5],
                          [-1.5, 1]])
    mean_b = np.array([-0.15, -0.15])
    cov_b = compute_cov_matrix(b_samples, mean_b)

    print(cov_r)
    print()
    print(cov_b)


