import numpy as np
from scipy.stats import multivariate_normal
from numpy import linalg as la
import heapq
import time


class ProbAndLabel(object):
    def __init__(self, label, prob):
        self.label = label
        self.prob = prob

    def __gt__(self, other):
        return self.prob > other.prob

    def __lt__(self, other):
        return self.prob < other.prob

    def __eq__(self, other):
        return self.prob == other.prob

    def __ne__(self, other):
        return self.prob != other.prob

    def __le__(self, other):
        return self.prob <= other.prob

    def __ge__(self, other):
        return self.prob >= other.prob


def mu_estimate_ml(data):
    return (1 / len(data)) * (sum(data))


def sigma_estimate_ml(data, mu):
    return (1 / len(data)) * (np.matmul(np.transpose(data - mu), (data - mu)))


def estimate_parameters_for_each_class(data, labels, classes):
    mu = np.zeros((len(classes), len(data[0])))
    sigma = np.zeros((len(classes), len(data[0]), len(data[0])))
    for i, class_ in enumerate(classes):
        class_data = []
        for j, sample in enumerate(data):
            if labels[j] == class_:
                class_data.append(sample)
        mu[i] = mu_estimate_ml(class_data)
        sigma[i] = sigma_estimate_ml(np.array(class_data), mu[i])
    return mu, sigma


def calc_prior_probabilities(labels):
    labels, counts = np.unique(labels, return_counts=True)
    return labels, counts / sum(counts)


def normalize_features(data):
    for i, sample in enumerate(data):
        data[i] = sample / max(sample)
    return data


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
    mu = mu_estimate_ml(new_data)
    sigma = sigma_estimate_ml(new_data, mu)
    return new_data, mu, sigma, to_remove_list


def classify_baysian(data, mus, sigmas, priors, labels):
    classes = np.zeros((len(data), 2))
    conf = np.zeros((len(data), 2))
    for i, sample in enumerate(data):
        probs = []
        for j, label in enumerate(labels):
            heapq.heappush(probs, ProbAndLabel(
                label,
                multivariate_normal.pdf(sample, mean=mus[j], cov=sigmas[j]) * priors[j])
                )
        heapq._heapify_max(probs)
        x = heapq.nlargest(2, probs)
        classes[i] = np.array([x[0].label, x[1].label])
        conf[i] = np.array([x[0].prob, x[1].prob])
    return classes, conf


def make_confusion_matrix(estimated_calsses, labels):
    classes = np.unique(labels)
    confusion_mat = np.zeros((len(classes), len(classes)))
    for estimate, label in zip(estimated_calsses, labels):
        confusion_mat[int(estimate[0])][int(label)] += 1
    print()
    print('confusion matrix:')
    print(confusion_mat)
    return confusion_mat


def make_confidence_matrix(estimated_classes, labels, confusion_mat, confs):
    classes = np.unique(labels)
    confidence_mat = np.zeros((len(classes), len(classes)))
    for estimate, label, conf in zip(estimated_classes, labels, confs):
        confidence_mat[int(estimate[0])][int(label)] += \
            (1 - (conf[1] / conf[0]))
    confidence_mat = confidence_mat / (confusion_mat + 1)
    print()
    print('confidence matrix:')
    print(confidence_mat)


def calc_accuracy(estimated_classes, labels, confs, duration):
    correct = 0
    for i, label in enumerate(labels):
        if label == estimated_classes[i][0]:
            correct += 1
    print()
    print('----------------------------')
    print('accuracy: ' + str(correct / len(labels)))
    print('duration: ', duration)
    confusion_mat = make_confusion_matrix(estimated_classes, labels)
    make_confidence_matrix(estimated_classes, labels, confusion_mat, confs)


def make_risk_coeffs_matrix(labels, landa_for_classes, landa_for_unknown):
    risk_coeffs = landa_for_classes * np.ones((len(labels)+1, len(labels)))
    for i in range(len(risk_coeffs)-1):
        risk_coeffs[i][i] = 0
    risk_coeffs[len(risk_coeffs)-1] = landa_for_unknown * np.ones(len(labels))
    return risk_coeffs


def calc_risk(data, j, mus, sigmas, priors, risk_coeffs):
    if j == len(risk_coeffs) - 1:
        return risk_coeffs[len(risk_coeffs)-1][0]
    risk = 0
    for i in range(len(risk_coeffs)-1):
        risk += risk_coeffs[j][i] * \
                multivariate_normal.pdf(data, mean=mus[i], cov=sigmas[i]) * priors[i]
    return risk


def classify_with_proper_risk(data, mus, sigmas, priors, labels):
    risky_labels = list(labels)
    risky_labels.append(len(labels))
    risk_coeffs = make_risk_coeffs_matrix(labels, 1, 0.8)
    classes = np.zeros(len(data))
    for i, sample in enumerate(data):
        min_risk = 10
        arg_min = -1
        for j, label in enumerate(risky_labels):
            risk = calc_risk(sample, j, mus, sigmas, priors, risk_coeffs)
            if risk < min_risk:
                min_risk = risk
                arg_min = j
        classes[i] = arg_min
    return classes


def calc_accuracy_respect_to_risks(estimated_classes, labels):
    correct = 0
    for i, label in enumerate(labels):
        if label == estimated_classes[i]:
            correct += 1
    data_size = 0
    for i in range(len(labels)):
        if estimated_classes[i] != max(labels):
            data_size += 1
    print()
    print('----------------------------')
    print('accuracy with respect to risks: ' + str(correct / data_size))


if __name__ == '__main__':

    train_data = np.genfromtxt('data/Reduced Fashion-MNIST/Train_Data.csv', delimiter=',')
    train_labels = np.genfromtxt('data/Reduced Fashion-MNIST/Train_Labels.csv', delimiter=',')
    labels, priors = calc_prior_probabilities(train_labels)

    train_data = normalize_features(train_data)

    mu = mu_estimate_ml(train_data)
    sigma = sigma_estimate_ml(train_data, mu)

    train_data, mu, sigma, removed_index = pca(train_data, sigma)

    test_data = np.genfromtxt('data/Reduced Fashion-MNIST/Test_Data.csv', delimiter=',')
    test_labels = np.genfromtxt('data/Reduced Fashion-MNIST/Test_Labels.csv', delimiter=',')

    test_data = normalize_features(test_data)
    test_data = remove_feature(test_data, removed_index)

    t1 = time.time()
    mus, sigmas = estimate_parameters_for_each_class(train_data, train_labels, labels)
    t2 = time.time()

    print('time to learn: ', t2-t1)

    np.set_printoptions(precision=2)

    t1 = time.time()
    estimated_classes, confs = classify_baysian(test_data, mus, sigmas, priors, labels)
    t2 = time.time()
    calc_accuracy(estimated_classes, test_labels, confs, t2-t1)

    estimated_classes_respect_to_risks = \
        classify_with_proper_risk(test_data, mus, sigmas, priors, labels)
    calc_accuracy_respect_to_risks(estimated_classes_respect_to_risks, test_labels)
