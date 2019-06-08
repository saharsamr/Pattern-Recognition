import numpy as np
from sklearn import svm
from time import time


def separate_two_classes(c1_indx, c2_indx, classes, data, labels):
    separated_data = [sample for sample, label in zip(data, labels) if
                      label == classes[c1_indx] or label == classes[c2_indx]]
    separated_labels = [label for sample, label in zip(data, labels) if
                        label == classes[c1_indx] or label == classes[c2_indx]]
    return separated_data, separated_labels


def majority_voting(ls):
    uniques = np.unique(ls)
    freq = np.zeros(len(uniques))
    for x in ls:
        for i, u in enumerate(uniques):
            if u == x:
                freq[i] += 1
                break
    return np.argmax(freq)


def train(data, labels):
    classes = np.unique(labels)
    svms = {}
    for i, c1 in enumerate(classes):
        for j in range(i+1, len(classes)):
            temp_train, temp_labels = separate_two_classes(i, j, classes, data, labels)
            svms[str(i)+', '+str(j)] = svm.SVC(kernel='poly', degree=6, coef0=1, gamma='scale')
            svms[str(i) + ', ' + str(j)].fit(temp_train, temp_labels)
    return svms


def test(svms, data, labels):
    classes = np.unique(labels)
    predictions = [[] for _ in data]
    for k, sample in enumerate(data):
        for i in range(len(classes)):
            for j in range(i+1, len(classes)):
                predictions[k].append(svms[str(i)+', '+str(j)].predict(sample.reshape(1, -1)))
    final_predictions = [predictions[i][majority_voting(predictions[i])][0] for i in range(len(data))]
    return final_predictions


def calc_ccr(predictions, labels):
    correct = 0
    for prediction, label in zip(predictions, labels):
        if prediction == label:
            correct += 1
    return correct / (len(labels))


if __name__ == "__main__":

    train_data = np.genfromtxt('../data/Reduced Fashion-MNIST/Train_Data.csv', delimiter=',')
    train_labels = np.genfromtxt('../data/Reduced Fashion-MNIST/Train_Labels.csv', delimiter=',')
    test_data = np.genfromtxt('../data/Reduced Fashion-MNIST/Test_Data.csv', delimiter=',')
    test_labels = np.genfromtxt('../data/Reduced Fashion-MNIST/Test_Labels.csv', delimiter=',')

    t1 = time()
    svms = train(train_data, train_labels)
    predictions = test(svms, train_data, train_labels)
    passed_time = time() - t1

    ccr = calc_ccr(predictions, train_labels)
    print('CCR: ', ccr)
    print('passed time: ', passed_time)
