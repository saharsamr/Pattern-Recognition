from sklearn.neighbors import KNeighborsClassifier, \
    RadiusNeighborsClassifier
import numpy as np


def normalize_features(data):
    for i, sample in enumerate(data):
        data[i] = sample / max(sample)
    return data


def calc_accuracy(estimated_classes, labels, print_label):
    correct = 0
    for estimate, label in zip(estimated_classes, labels):
        if estimate == label:
            correct += 1
    print('accuracy for ' + print_label + ': ' + str(correct / len(labels)))


if __name__ == '__main__':

    train_data = np.genfromtxt('../data/Reduced Fashion-MNIST/Train_Data.csv', delimiter=',')
    train_labels = np.genfromtxt('../data/Reduced Fashion-MNIST/Train_Labels.csv', delimiter=',')
    test_data = np.genfromtxt('../data/Reduced Fashion-MNIST/Test_Data.csv', delimiter=',')
    test_labels = np.genfromtxt('../data/Reduced Fashion-MNIST/Test_Labels.csv', delimiter=',')

    train_data = normalize_features(train_data)
    test_data = normalize_features(test_data)

    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(train_data, train_labels)
    estimated_classes = knn_classifier.predict(test_data)
    calc_accuracy(estimated_classes, test_labels, 'KNN Classifier with k = 3')

    parzen_classifier = RadiusNeighborsClassifier(radius=8.5)
    parzen_classifier.fit(train_data, train_labels)
    estimated_classes = parzen_classifier.predict(test_data)
    calc_accuracy(estimated_classes, test_labels, 'Parzen Estimator')
