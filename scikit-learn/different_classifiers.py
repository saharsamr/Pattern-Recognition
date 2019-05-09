from sklearn.neighbors import KNeighborsClassifier, \
    RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from numpy import linalg as la


def mu_estimate_ml(data):
    return (1 / len(data)) * (sum(data))


def sigma_estimate_ml(data, mu):
    return (1 / len(data)) * (np.matmul(np.transpose(data - mu), (data - mu)))


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

    mu = mu_estimate_ml(train_data)
    sigma = sigma_estimate_ml(train_data, mu)

    train_data = normalize_features(train_data)
    train_data, removed_indices = pca(normalize_features(train_data), sigma)
    test_data = remove_feature(test_data, removed_indices)

    parzen_classifier = RadiusNeighborsClassifier(radius=2.8)
    parzen_classifier.fit(train_data, train_labels)
    estimated_classes = parzen_classifier.predict(test_data)
    calc_accuracy(estimated_classes, test_labels, 'Parzen Estimator')

    gnb = GaussianNB()
    gnb.fit(train_data, train_labels)
    estimated_calsses = gnb.predict(test_data)
    calc_accuracy(estimated_calsses, test_labels, 'Gaussian Naive Bayes')
