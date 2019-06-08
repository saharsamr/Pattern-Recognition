import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV


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

    svc = svm.SVC(kernel='rbf')
    parameters = {'C': [1, 5], 'gamma': [0.1, 0.9]}
    clf = GridSearchCV(svc, parameters, cv=5)
    clf.fit(train_data, train_labels)

    predictions = clf.predict(test_data)

    ccr = calc_ccr(predictions, test_labels)
    print('CCR', ccr)
    print('parameters: ' + str(clf.get_params()))


