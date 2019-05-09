import numpy as np
from math import sqrt
import heapq


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


def separate_data_by_classes(data, labels, classes):
    separated_data = [[] for _ in classes]
    for i, class_ in enumerate(classes):
        for sample, label in zip(data, labels):
            if label == class_:
                separated_data[int(class_)].append(sample)
    return separated_data


def distance(query, sample):
    return sqrt(sum((query - sample) ** 2))


def min_mean_dist_for_point(query, separated_data):
    mean_distance = []
    for i, cluster in enumerate(separated_data):
        distances = 0
        for sample in cluster:
            distances += distance(query, sample)
        heapq.heappush(mean_distance, LabelAndDistance(i, distances / len(cluster)))
    return heapq.heappop(mean_distance).label


def classify_min_mean_distance(queries, separated_data):
    estimated_classes = np.zeros(len(queries))
    for i, query in enumerate(queries):
        estimated_classes[i] = min_mean_dist_for_point(query, separated_data)
    return estimated_classes


def make_confusion_matrix(estimated_calsses, labels):
    classes = np.unique(labels)
    confusion_mat = np.zeros((len(classes), len(classes)))
    for estimate, label in zip(estimated_calsses, labels):
        confusion_mat[int(estimate)][int(label)] += 1
    print()
    print(confusion_mat)


def calc_accuracy(estimated_classes, labels):
    correct = 0
    for estimate, label in zip(estimated_classes, labels):
        if estimate == label:
            correct += 1
    print()
    print('----------------------------')
    print('accuracy: ' + str(correct / len(labels)))
    make_confusion_matrix(estimated_classes, labels)


if __name__ == '__main__':

    train_data = np.genfromtxt('./../data/Reduced Fashion-MNIST/Train_Data.csv', delimiter=',')
    train_labels = np.genfromtxt('./../data/Reduced Fashion-MNIST/Train_Labels.csv', delimiter=',')

    labels = np.unique(train_labels)
    separated_data = separate_data_by_classes(train_data, train_labels, labels)

    test_data = np.genfromtxt('./../data/Reduced Fashion-MNIST/Test_Data.csv', delimiter=',')
    test_labels = np.genfromtxt('./../data/Reduced Fashion-MNIST/Test_Labels.csv', delimiter=',')

    estimated_classes = classify_min_mean_distance(test_data[0:500], separated_data)
    calc_accuracy(estimated_classes, test_labels[0:500])
