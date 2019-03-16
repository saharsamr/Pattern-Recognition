import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.stats import multivariate_normal
from scipy.stats import norm


def plot_normal_dist():
    x = np.arange(-10, 10, 0.0001)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, 0.5*norm.pdf(x, 1, 1), label='N(1,1)')
    ax.plot(x, 0.5*norm.pdf(x, -1, 1), label='N(-1,1)')
    ax.plot(x, 0.75*0.5*(norm.pdf(x, 1, 1)+norm.pdf(x, -1, 1)), label='unknown class')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    plot_normal_dist()

