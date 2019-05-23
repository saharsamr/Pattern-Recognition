from numpy import genfromtxt
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def separate_data_by_classes(data, labels):
    classes = np.unique(labels)
    separated_data = [[] for _ in classes]
    for i, class_ in enumerate(classes):
        for sample, label in zip(data, labels):
            if label == class_:
                separated_data[int(class_)-1].append(sample)
    return separated_data


def phi(sample):
    return np.array([sample[0]**2, sample[1]**2, sample[0]*sample[1]]) + 1000


def apply_kernel(sample, data):
    return sum([np.matmul(phi(sample), phi(i)) for i in data])


if __name__ == "__main__":
    data = np.transpose(genfromtxt('../data/haberman.data', delimiter=','))
    labels = data[3]
    data = np.transpose(data[0:3])

    pca = PCA(n_components=2)
    new_data = pca.fit_transform(data)

    separated_data = separate_data_by_classes(new_data, labels)

    fig = plt.figure('original data')

    plt.scatter([i[0] for i in separated_data[0]], [i[1] for i in separated_data[0]])
    plt.scatter([i[0] for i in separated_data[1]], [i[1] for i in separated_data[1]])

    fig = plt.figure('transmitted data')
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter([i[0] for i in separated_data[0]],
               [i[1] for i in separated_data[0]],
               [apply_kernel(sample, separated_data[0]) for sample in separated_data[0]])
    ax.scatter([i[0] for i in separated_data[1]],
               [i[1] for i in separated_data[1]],
               [apply_kernel(sample, separated_data[1]) for sample in separated_data[1]])

    plt.show()
