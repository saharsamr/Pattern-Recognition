import numpy as np
from math import sqrt
import heapq
from numpy import linalg as la


class LabelAndDistance(object):
    def __init__(self, label, distance):
        self.label = label
        self.distance = distance

    def __gt__(self, other):
        return self.distance > other.distance

    def __lt__(self, other):
        return self.distance < other.distance

    def __eq__(self, other):
        return self.distance == other.distance

    def __ne__(self, other):
        return self.distance != other.distance

    def __le__(self, other):
        return self.distance <= other.distance

    def __ge__(self, other):
        return self.distance >= other.distance


def mu_estimate_ml(data):
    return (1 / len(data)) * (sum(data))


def sigma_estimate_ml(data, mu):
    return (1 / len(data)) * (np.matmul(np.transpose(data - mu), (data - mu)))


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


def distance(query, sample):
    return sqrt(sum((query - sample) ** 2))


def majority_voting(majority):
    frequencies = {}
    for element in majority:
        if element in frequencies.keys():
            frequencies[element] += 1
        else:
            frequencies[element] = 1
    print(max(frequencies, key=frequencies.get))
    return max(frequencies, key=frequencies.get)


def knn_for_single_point(query, data, labels, k):
    distances = []
    for sample, label in zip(data, labels):
        heapq.heappush(distances, LabelAndDistance(label, distance(query, sample)))
    k_mins = []
    for _ in range(k):
        k_mins.append(heapq.heappop(distances))
    majority = [x.label for x in k_mins]
    return majority_voting(majority)


def classify_knn(queries, data, labels, k):
    estimated_classes = np.zeros(len(queries))
    for i, query in enumerate(queries):
        estimated_classes[i] = knn_for_single_point(query, data, labels, k)
    return estimated_classes


def make_confusion_matrix(estimated_calsses, labels):
    classes = np.unique(labels)
    confusion_mat = np.zeros((len(classes), len(classes)))
    for estimate, label in zip(estimated_calsses, labels):
        confusion_mat[int(estimate)][int(label)] += 1
    print(confusion_mat)


def calc_accuracy(estimated_classes, labels):
    correct = 0
    for estimate, label in zip(estimated_classes, labels):
        if estimate == label:
            correct += 1
    print('----------------------------')
    print('accuracy: ' + str(correct / len(labels)))
    make_confusion_matrix(estimated_classes, labels)


if __name__ == "__main__":
    train_data = np.genfromtxt('./../data/Reduced Fashion-MNIST/Train_Data.csv', delimiter=',')
    train_labels = np.genfromtxt('./../data/Reduced Fashion-MNIST/Train_Labels.csv', delimiter=',')

    mu = mu_estimate_ml(train_data)
    sigma = sigma_estimate_ml(train_data, mu)

    train_data = normalize_features(train_data)
    train_data, removed_indices = pca(normalize_features(train_data), sigma)

    labels = np.unique(train_labels)
    separated_data = separate_data_by_classes(train_data, train_labels, labels)

    test_data = np.genfromtxt('./../data/Reduced Fashion-MNIST/Test_Data.csv', delimiter=',')
    test_labels = np.genfromtxt('./../data/Reduced Fashion-MNIST/Test_Labels.csv', delimiter=',')

    test_data = normalize_features(test_data)
    test_data = remove_feature(test_data, removed_indices)

    estimated_classes = classify_knn(test_data[0:500], train_data, train_labels, 3)
    calc_accuracy(estimated_classes, test_labels[0:500])
