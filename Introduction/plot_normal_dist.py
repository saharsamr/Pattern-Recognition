import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.stats import multivariate_normal


def plot_normal_dist(mu1, cov1, mu2, cov2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.mgrid[-5.0:5.0:50j, -5.0:5.0:50j]
    xy = np.column_stack([x.flat, y.flat])
    z1 = (1/3)*multivariate_normal.pdf(xy, mean=mu1, cov=cov1)
    z1 = z1.reshape(x.shape)
    z2 = multivariate_normal.pdf(xy, mean=mu2, cov=cov2)
    z2 = (2/3)*z2.reshape(x.shape)
    ax.plot_surface(x, y, np.where(z1 < z2, z1, np.nan), label='red points')
    ax.plot_surface(x, y, z2)
    ax.plot_surface(x, y, np.where(z1 >= z2, z1, np.nan), label='black points')
    # plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    plot_normal_dist([1.33, 1.61], [[0.6250125, 0.2083375], [0.2083375, 1.1111125]],
                     [-0.15, -0.15], [[1.725, 0.00277778], [0.00277778, 0.55833333]])

