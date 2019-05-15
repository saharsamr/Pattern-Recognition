from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt


if __name__ == '__main__':

    fashion_mn_data = MNIST('../data/Fashion-MNIST')
    train_data, train_labels = fashion_mn_data.load_training()
    test_data, test_labels = fashion_mn_data.load_testing()

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    pca_s = []
    for i in range(0, 10):
        print(i)
        pca = PCA(n_components=i * 70 + 9)
        new_train_data = pca.fit_transform(train_data)
        new_test_data = pca.transform(test_data)

        gnb = GaussianNB()
        gnb.fit(new_train_data, train_labels)
        predicted_labels = gnb.predict(new_test_data)
        pca_s.append(accuracy_score(test_labels, predicted_labels))

    lda_s = []
    for i in range(0, 10):
        print(i)
        lda = LDA(n_components=i * 70 + 9)
        train_data = lda.fit_transform(train_data, train_labels)
        test_data = lda.transform(test_data)

        gnb = GaussianNB()
        gnb.fit(train_data, train_labels)
        predicted_labels = gnb.predict(test_data)
        lda_s.append(accuracy_score(test_labels, predicted_labels))

    plt.figure('result')
    plt.plot([i * 70 + 9 for i in range(0, 10)], pca_s, label='pca')
    plt.plot([i * 70 + 9 for i in range(0, 10)], lda_s, label='lda')
    plt.legend()
    plt.show()

