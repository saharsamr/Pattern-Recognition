import numpy as np
from mnist import MNIST
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import heapq
import matplotlib.pyplot as plt

class DiscriminationOfFeatures(object):
    def __init__(self, feature_index, discrimination):
        self.feature_index = feature_index
        self.discrimination = discrimination

    def __ge__(self, other):
        return self.discrimination >= other.discrimination

    def __le__(self, other):
        return self.discrimination <= other.discrimination

    def __ne__(self, other):
        return self.discrimination != other.discrimination

    def __eq__(self, other):
        return self.discrimination == other.discrimination

    def __lt__(self, other):
        return self.discrimination < other.discrimination

    def __gt__(self, other):
        return self.discrimination > other.discrimination


def report_gnb_accuracy(train_data, train_labels, test_data, test_labels):
    gnb = GaussianNB()
    gnb.fit(train_data, train_labels)
    predicted_labels = gnb.predict(test_data)
    return accuracy_score(test_labels, predicted_labels)


def feature_selection(
        train_data,
        train_labels,
        test_data,
        test_labels,
        pre_selected_features,
        candidate_features
):
    features_sorted = []
    for feature in candidate_features:
        feature_indices = np.array([*pre_selected_features, feature.feature_index], dtype=np.int)
        discrimination = report_gnb_accuracy(
            train_data[:, feature_indices],
            train_labels,
            test_data[:, feature_indices],
            test_labels
        )
        features_sorted.append(DiscriminationOfFeatures(feature.feature_index, discrimination))
    new_feature = heapq.nlargest(1, features_sorted)[0]
    pre_selected_features.append(new_feature.feature_index)
    return pre_selected_features, new_feature.discrimination


def forward_selection(train_data, train_labels, test_data, test_labels):
    candidate_features = [DiscriminationOfFeatures(i, 0) for i in range(len(train_data[0]))]
    max_discrimination = 0
    pre_selected_features = []
    num_of_features_and_ccr = []
    best_features = []
    for i in range(len(train_data[0])):
        pre_selected_features, discrimination = \
            feature_selection(
                train_data, train_labels,
                test_data, test_labels,
                pre_selected_features, candidate_features
            )
        if max_discrimination < discrimination:
            best_features = pre_selected_features
            max_discrimination = discrimination
        num_of_features_and_ccr.append((len(pre_selected_features), discrimination))
        print(discrimination)
    return best_features, num_of_features_and_ccr


if __name__ == '__main__':
    fashion_mn_data = MNIST('../data/Fashion-MNIST')
    train_data, train_labels = fashion_mn_data.load_training()
    test_data, test_labels = fashion_mn_data.load_testing()

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    best_features, feature_num_and_ccr = \
        forward_selection(train_data, train_labels, test_data[0:500], test_labels[0:500])

    plt.plot([point[0] for point in feature_num_and_ccr],
             [point[1] for point in feature_num_and_ccr])
