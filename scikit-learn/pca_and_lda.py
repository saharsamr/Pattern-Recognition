from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
from mnist import MNIST


if __name__ == '__main__':

    fashion_mn_data = MNIST('../data/Fashion-MNIST')
    train_data, train_labels = fashion_mn_data.load_training()
    test_data, test_labels = fashion_mn_data.load_testing()

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    pca = PCA(n_components=200)
    new_train_data = pca.fit_transform(train_data)
    new_test_data = pca.transform(test_data)

    gnb = GaussianNB()
    gnb.fit(new_train_data, train_labels)
    predicted_labels = gnb.predict(new_test_data)
    print(accuracy_score(test_labels, predicted_labels))

    lda = LDA()
    train_data = lda.fit_transform(train_data, train_labels)
    test_data = lda.transform(test_data)

    gnb = GaussianNB()
    gnb.fit(train_data, train_labels)
    predicted_labels = gnb.predict(test_data)
    print(accuracy_score(test_labels, predicted_labels))
