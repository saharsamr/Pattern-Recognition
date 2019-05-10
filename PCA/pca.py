import numpy as np
from numpy import linalg as la
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def mu_estimate_ml(data):
    return (1 / len(data)) * (sum(data))


def sigma_estimate_ml(data, mu):
    return (1 / len(data)) * (np.matmul(np.transpose(data - mu), (data - mu)))


def pca(data, n_dims):
    mu = mu_estimate_ml(data)
    sigma = sigma_estimate_ml(data, mu)
    eigenvalues, eigenvectors = la.eig(sigma)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[sorted_indices]
    return eigenvectors[0:n_dims+1], eigenvalues


def apply_transformation(transform, data):
    new_data = np.zeros((len(data), len(transform)))
    for i, row in enumerate(data):
        new_data[i] = np.matmul(transform, row)
    return new_data


if __name__ == "__main__":
    fashion_mn_data = MNIST('../data/Fashion-MNIST')
    train_data, train_labels = fashion_mn_data.load_training()
    test_data, test_labels = fashion_mn_data.load_testing()

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    transform, eigenvalues = pca(train_data, 200)

    reduced_train_data = apply_transformation(transform, train_data)
    reduced_test_data = apply_transformation(transform, test_data)

    gnb = GaussianNB()
    gnb.fit(reduced_train_data, train_labels)
    predicted_labels = gnb.predict(reduced_test_data)
    print('with PCA: ', accuracy_score(test_labels, predicted_labels))

    gnb = GaussianNB()
    gnb.fit(train_data, train_labels)
    predicted_labels = gnb.predict(test_data)
    print('without PCA: ', accuracy_score(test_labels, predicted_labels))

    plt.plot(eigenvalues)
    plt.show()
