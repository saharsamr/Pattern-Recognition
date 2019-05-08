import numpy as np
from scipy.stats import multivariate_normal


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


def classify_parzen(query, data, parzen_window, h):
    parzen_sigma = 0
    for sample in data:
        parzen_sigma += parzen_window(query, sample, h)
    return parzen_sigma / (len(data) * h ** len(data[0]))


def rectangular_parzen_window(query, sample, h):
    for i, dim in enumerate(sample):
        if abs(query[i] - dim) / h > 0.5:
            return 0
    return 1


def gaussian_parzen_window(query, sample, h):
    return multivariate_normal.pdf((query - sample) / h,
                                   mean=np.zeros(len(query)),
                                   cov=np.identity(len(query)))


if __name__ == '__main__':

    train_data = np.genfromtxt('data/Reduced Fashion-MNIST/Train_Data.csv', delimiter=',')
    train_labels = np.genfromtxt('data/Reduced Fashion-MNIST/Train_Labels.csv', delimiter=',')

    labels = np.unique(train_labels)
    separated_data = separate_data_by_classes(train_data, train_labels, labels)
