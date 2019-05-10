import numpy as np
from mnist import MNIST
from PCA.pca import pca, apply_transformation
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from numpy import linalg as la


def mu_estimate(data):
    return (1 / len(data)) * (sum(data))


def scatter_estimate(data, mu):
    return (1 / len(data)) * (np.matmul(np.transpose(data - mu), (data - mu)))


def separate_data_by_classes(data, labels, classes):
    separated_data = [[] for _ in classes]
    for i, class_ in enumerate(classes):
        for sample, label in zip(data, labels):
            if label == class_:
                separated_data[int(class_)].append(sample)
    return separated_data


def whitening_pca(train_data, test_data, n_dims):
    transform, eigenvalues = pca(train_data, n_dims)
    train_data = apply_transformation(transform, train_data)
    print(train_data.shape)
    print(eigenvalues.shape)
    eigenvalues = eigenvalues[0:len(train_data[0])]
    train_data = train_data / eigenvalues
    test_data = apply_transformation(transform, test_data)
    test_data = test_data / eigenvalues
    return train_data, test_data


def calc_within_scatter_matrix(separated_data):
    s_w = np.zeros((len(separated_data[0][0]), len(separated_data[0][0])))
    for cluster in separated_data:
        mu = mu_estimate(cluster)
        s_w += scatter_estimate(cluster, mu)
    return s_w


def calc_between_scatter_matrix(separated_data):
    mus = np.array([mu_estimate(cluster) for cluster in separated_data])
    mu = sum([len(cluster) * mus[i] for i, cluster in enumerate(separated_data)]) / \
         sum([len(cluster) for cluster in separated_data])
    s_b = np.zeros((len(separated_data[0][0]), len(separated_data[0][0])))
    for mu_i, cluster in zip(mus, separated_data):
        s_b += len(cluster) * np.matmul(mu_i - mu, np.transpose(mu_i - mu))
    return s_b


def lda(separated_data):
    s_w = calc_within_scatter_matrix(separated_data)
    s_b = calc_between_scatter_matrix(separated_data)
    s_p = np.matmul(np.linalg.inv(s_w), s_b)
    eigenvalues, eigenvectors = la.eig(s_p)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[sorted_indices]
    rank = min(len(s_w[0]), len(separated_data)-1)
    eigenvectors = eigenvectors[0:rank]
    return eigenvectors


if __name__ == '__main__':
    fashion_mn_data = MNIST('../data/Fashion-MNIST')
    train_data, train_labels = fashion_mn_data.load_training()
    test_data, test_labels = fashion_mn_data.load_testing()

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    train_data, test_data = whitening_pca(train_data, test_data, 100)

    classes = np.unique(train_labels)
    separated_data = separate_data_by_classes(train_data, train_labels, classes)

    transform = lda(separated_data)

    train_data = apply_transformation(transform, train_data)
    test_data = apply_transformation(transform, test_data)

    gnb = GaussianNB()
    gnb.fit(train_data, train_labels)
    predicted_labels = gnb.predict(test_data)
    print('accuracy: ', accuracy_score(test_labels, predicted_labels))
