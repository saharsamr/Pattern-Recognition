import numpy as np
from sklearn import svm


def separate_two_classes(c1_indx, c2_indx, classes, data, labels):
    separated_data = [sample for sample, label in zip(data, labels) if
                      label == classes[c1_indx] or label == classes[c2_indx]]
    separated_labels = [label for sample, label in zip(data, labels) if
                        label == classes[c1_indx] or label == classes[c2_indx]]
    return separated_data, separated_labels


def train(data, labels):
    classes = np.unique(labels)
    svms = {}
    for i, c1 in enumerate(classes):
        for j in range(i+1, len(classes)):
            temp_train, temp_labels = separate_two_classes(i, j, classes, data, labels)
            svms[str(i)+', '+str(j)] = svm.SVC(kernel='poly', degree=4, coef0=1, gamma='scale')
            svms[str(i) + ', ' + str(j)].fit(temp_train, temp_labels)
    return svms


if __name__ == "__main__":

    train_data = np.genfromtxt('../data/Reduced Fashion-MNIST/Train_Data.csv', delimiter=',')
    train_labels = np.genfromtxt('../data/Reduced Fashion-MNIST/Train_Labels.csv', delimiter=',')
    test_data = np.genfromtxt('../data/Reduced Fashion-MNIST/Test_Data.csv', delimiter=',')
    test_labels = np.genfromtxt('../data/Reduced Fashion-MNIST/Test_Labels.csv', delimiter=',')

    svms = train(train_data, train_labels)
