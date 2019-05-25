from numpy import genfromtxt
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import svm
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
    return sum([np.matmul(phi(sample), phi(i)) for i in data])/1000000


def weight_derivative(weight, data, labels, landa):
    d_w = np.zeros(len(weight))
    for i, w in enumerate(weight):
        d_w[i] = landa * w - sum([label * x[i] for label, x in zip(labels, data)])
    return d_w


def find_hyperplane(data, labels, landa, alpha):
    weights = np.random.rand(len(data[0]))
    for _ in range(10000):
        weights -= alpha * weight_derivative(weights, data, labels, landa)
    return weights


def increase_dim(data, labels):
    transformed = np.zeros((len(data), 3))
    for i in range(len(data)):
        transformed[i] = np.array([data[i][0], data[i][1],
                                   apply_kernel(data[i], separated_data[int(labels[i])-1])])
    return transformed


def plot_surface(weights, ax, d=1350000000):
    xx, yy = np.meshgrid(range(-50, 50), range(-50, 50))
    z = (d - xx * weights[0] - yy * weights[1]) / weights[2]
    ax.plot_surface(xx, yy, z)


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

    new_data = increase_dim(new_data, labels)
    separated_data = separate_data_by_classes(new_data, labels)

    ax.scatter([i[0] for i in separated_data[0]],
               [i[1] for i in separated_data[0]],
               [i[2] for i in separated_data[0]])
    ax.scatter([i[0] for i in separated_data[1]],
               [i[1] for i in separated_data[1]],
               [i[2] for i in separated_data[1]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    weights = find_hyperplane(new_data, labels, 0.1, 0.1)

    plot_surface(weights, ax)

    svc = svm.SVC(gamma='scale')
    svc.fit(new_data, labels)

    plot_surface(svc._get_coef()[0], ax, d=-600000)

    plt.show()
