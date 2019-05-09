import numpy as np
from scipy.stats import multivariate_normal
from numpy import linalg as la


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
    classes = np.zeros(len(data))
    for i, sample in enumerate(data):
        max_prob = 0
        arg_max = -1
        for j, label in enumerate(labels):
            prob = multivariate_normal.pdf(sample, mean=mus[j], cov=sigmas[j]) * priors[j]
            if prob > max_prob:
                max_prob = prob
                arg_max = j
        classes[i] = arg_max
    return classes


def make_confusion_matrix(estimated_calsses, labels):
    classes = np.unique(labels)
    confusion_mat = np.zeros((len(classes), len(classes)))
    for estimate, label in zip(estimated_calsses, labels):
        confusion_mat[int(estimate)][int(label)] += 1
    print(confusion_mat)


def calc_accuracy(estimated_classes, labels):
    correct = 0
    for i, label in enumerate(labels):
        if label == estimated_classes[i]:
            correct += 1
    print('----------------------------')
    print('accuracy: ' + str(correct / len(labels)))
    make_confusion_matrix(estimated_classes, labels)


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
        if estimated_classes[i] != len(labels):
            data_size += 1
    risky_labels = list(labels)
    risky_labels.append(len(labels))
    print('accuracy with respect to risks: ' + str(correct / data_size))
    make_confusion_matrix(estimated_classes, risky_labels)


if __name__ == '__main__':

    train_data = np.genfromtxt('data/Reduced Fashion-MNIST/Train_Data.csv', delimiter=',')
    train_labels = np.genfromtxt('data/Reduced Fashion-MNIST/Train_Labels.csv', delimiter=',')
    labels, priors = calc_prior_probabilities(train_labels)

    train_data = normalize_features(train_data)

    mu = mu_estimate_ml(train_data)
    sigma = sigma_estimate_ml(train_data, mu)

    train_data, mu, sigma, removed_index = pca(train_data, sigma)

    mus, sigmas = estimate_parameters_for_each_class(train_data, train_labels, labels)

    test_data = np.genfromtxt('data/Reduced Fashion-MNIST/Test_Data.csv', delimiter=',')
    test_labels = np.genfromtxt('data/Reduced Fashion-MNIST/Test_Labels.csv', delimiter=',')

    test_data = normalize_features(test_data)
    test_data = remove_feature(test_data, removed_index)

    estimated_classes = classify_baysian(test_data, mus, sigmas, priors, labels)
    calc_accuracy(estimated_classes, test_labels)

    estimated_classes_respect_to_risks = \
        classify_with_proper_risk(test_data, mus, sigmas, priors, labels)
    calc_accuracy_respect_to_risks(estimated_classes_respect_to_risks, test_labels)
