import numpy as np
from sklearn import svm


def separate_two_classes(c, labels):
    new_labels = np.zeros(len(labels))
    for i, label in enumerate(labels):
        if label == c:
            new_labels[i] = c
        else:
            new_labels[i] = -1
    return new_labels


def majority_voting(ls, classes):
    freq = np.zeros(len(classes))
    for x in ls:
        for i, c in enumerate(classes):
            if c == x:
                freq[i] += 1
                break
    return np.argmax(freq)


def train(data, labels):
    classes = np.unique(labels)
    svms = {}
    for i, c in enumerate(classes):
        temp_labels = separate_two_classes(c, labels)
        svms[str(i)] = svm.SVC(kernel='poly', degree=4, coef0=1, gamma='scale')
        svms[str(i)].fit(data, temp_labels)
    return svms


def test(svms, data, labels):
    classes = np.unique(labels)
    predictions = [[] for _ in data]
    for k, sample in enumerate(data):
        for i, c in enumerate(classes):
            predictions[k].append(svms[str(i)].predict(sample.reshape(1, -1)))
    final_predictions = [predictions[i][majority_voting(predictions[i], classes)][0]
                         for i in range(len(data))]
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

    svms = train(train_data, train_labels)
    predictions = test(svms, train_data, train_labels)

    ccr = calc_ccr(predictions, train_labels)
    print(ccr)
