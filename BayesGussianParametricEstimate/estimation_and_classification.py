import numpy as np


def mu_estimate_ml(data):
    return (1 / len(data)) * (sum(data))


def sigma_estimate_ml(data, mu):
    return (1 / len(data)) * (np.matmul(np.transpose(data - mu), (data - mu)))


if __name__ == '__main__':

    train_data = np.genfromtxt('data/Reduced Fashion-MNIST/Train_Data.csv', delimiter=',')
    train_label = np.genfromtxt('data/Reduced Fashion-MNIST/Train_Labels.csv', delimiter=',')

    mu = mu_estimate_ml(train_data)
    sigma = sigma_estimate_ml(train_data, mu)


