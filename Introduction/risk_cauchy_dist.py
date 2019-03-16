import numpy as np
from scipy.stats import cauchy
import matplotlib.pyplot as plt


if __name__ == "__main__":
    plt.xlim(-20, 20)
    x = np.linspace(cauchy.ppf(0.0001), cauchy.ppf(0.9999), 10000)
    plt.plot(x, 0.5*cauchy.pdf(x, loc=3, scale=1), 'r-', lw=3, alpha=0.6, label='g2(x)')
    plt.plot(x, 2*0.5*cauchy.pdf(x, loc=5, scale=1), 'b-', lw=3, alpha=0.6, label='g1(x)')
    plt.legend(loc='best')
    plt.show()
