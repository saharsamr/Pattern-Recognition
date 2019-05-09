import numpy as np
from scipy.stats import multivariate_normal
from numpy import linalg as la
from math import sqrt, pi, exp


def mu_estimate_ml(data):
    return (1 / len(data)) * (sum(data))


def sigma_estimate_ml(data, mu):
    return (1 / len(data)) * (np.matmul(np.transpose(data - mu), (data - mu)))


def calc_prior_probabilities(labels):
    labels, counts = np.unique(labels, return_counts=True)
    return labels, counts / sum(counts)


def separate_data_by_classes(data, labels, classes):
    separated_data = [[] for _ in classes]
    for i, class_ in enumerate(classes):
        for sample, label in zip(data, labels):
            if label == class_:
                separated_data[int(class_)].append(sample)
    return separated_data


def normalize_features(data):
    for i, sample in enumerate(data):
        data[i] = sample / max(sample)
    return data


def find_to_remove_features(sigma):
    eigen_vectors = la.eig(sigma)[1]
    D = np.matmul(np.matmul(np.transpose(eigen_vectors), sigma), eigen_vectors)
    D = np.diag(np.diag(D))
    to_remove_list = []
    for i in range(len(D)):
        if D[i][i] < 0.05:
            to_remove_list.append(i)
    return to_remove_list


def remove_feature(data, to_remove_indices):
    new_data = np.zeros((len(data), len(data[0])-len(to_remove_indices)))
    for i, sample in enumerate(data):
        new_data[i] = np.delete(sample, to_remove_indices)
    return new_data


def pca(data, sigma):
    to_remove_list = find_to_remove_features(sigma)
    new_data = remove_feature(data, to_remove_list)
    return new_data, to_remove_list


def calc_parzen_for_point(query, data, parzen_window, h):
    parzen_sigma = 0
    for sample in data:
        parzen_sigma += parzen_window(query, sample, h)
    return parzen_sigma / (len(data) * (h ** len(data[0])))


def rectangular_parzen_window(query, sample, h):
    for i, dim in enumerate(sample):
        if abs(query[i] - dim) / h > 0.5:
            return 0
    return 10 / 6


def gaussian_parzen_window(query, sample, h):
    result = 1
    for i, dim in enumerate(sample):
        # result *= multivariate_normal.pdf((query[i] - dim), mean=0, cov=1)
        result *= (1 / sqrt(2 * pi)) * (exp(-1 * ((query[i] - dim) ** 2) / (2 * h)))
    return result


def classify_parzen(data, separated_data, parzen_window):
    classes = np.zeros(len(data))
    for j, sample in enumerate(data):
        max_parzen = 0
        arg_max = -1
        for i, cluster in enumerate(separated_data):
            parzen = calc_parzen_for_point(sample, cluster, parzen_window, 1)
            if parzen > max_parzen:
                max_parzen = parzen
                arg_max = i
        classes[j] = arg_max
        print(max_parzen)
    return classes


def calc_accuracy(estimated_classes, labels):
    correct = 0
    for label, estimate in zip(estimated_classes, labels):
        if label == estimate:
            correct += 1
    print('accuracy: ' + str(correct / len(labels)))


if __name__ == '__main__':

    train_data = np.genfromtxt('data/Reduced Fashion-MNIST/Train_Data.csv', delimiter=',')
    train_labels = np.genfromtxt('data/Reduced Fashion-MNIST/Train_Labels.csv', delimiter=',')

    mu = mu_estimate_ml(train_data)
    sigma = sigma_estimate_ml(train_data, mu)

    train_data = normalize_features(train_data)
    train_data, removed_indices = pca(normalize_features(train_data), sigma)

    labels = np.unique(train_labels)
    separated_data = separate_data_by_classes(train_data, train_labels, labels)

    test_data = np.genfromtxt('data/Reduced Fashion-MNIST/Test_Data.csv', delimiter=',')
    test_labels = np.genfromtxt('data/Reduced Fashion-MNIST/Test_Labels.csv', delimiter=',')

    test_data = normalize_features(test_data)
    test_data = remove_feature(test_data, removed_indices)

    estimated_classes = classify_parzen(test_data, separated_data, rectangular_parzen_window)
    calc_accuracy(estimated_classes, test_labels)

    estimated_classes = classify_parzen(test_data, separated_data, gaussian_parzen_window)
    calc_accuracy(estimated_classes, test_labels)
