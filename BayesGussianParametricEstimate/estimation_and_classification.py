import numpy as np
from scipy.stats import multivariate_normal


def mu_estimate_ml(data):
    return (1 / len(data)) * (sum(data))


def sigma_estimate_ml(data, mu):
    return (1 / len(data)) * (np.matmul(np.transpose(data - mu), (data - mu)))


def calc_prior_probabilities(labels):
    labels, counts = np.unique(labels, return_counts=True)
    return labels, counts / sum(counts)


def classify_baysian(data, mu, sigma, priors, labels):
    classes = np.zeros(len(data))
    for i, sample in enumerate(data):
        max_prob = 0
        arg_max = -1
        for j, label in enumerate(labels):
            prob = multivariate_normal.pdf(sample, mean=mu, cov=sigma) * priors[j]
            if prob > max_prob:
                max_prob = prob
                arg_max = j
        print(arg_max)
        classes[i] = arg_max
    return classes


if __name__ == '__main__':

    train_data = np.genfromtxt('data/Reduced Fashion-MNIST/Train_Data.csv', delimiter=',')
    train_labels = np.genfromtxt('data/Reduced Fashion-MNIST/Train_Labels.csv', delimiter=',')

    mu = mu_estimate_ml(train_data)
    sigma = sigma_estimate_ml(train_data, mu)

    labels, priors = calc_prior_probabilities(train_labels)

    test_data = np.genfromtxt('data/Reduced Fashion-MNIST/Test_Data.csv', delimiter=',')
    test_labels = np.genfromtxt('data/Reduced Fashion-MNIST/Test_Labels.csv', delimiter=',')

    estimated_classes = classify_baysian(test_data, mu, sigma, priors, labels)
    print(estimated_classes)


