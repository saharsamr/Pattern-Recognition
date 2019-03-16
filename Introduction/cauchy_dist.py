import numpy as np
from scipy.stats import cauchy
import matplotlib.pyplot as plt


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1)
    plt.xlim(-20, 20)
    x = np.linspace(cauchy.ppf(0.0001), cauchy.ppf(0.9999), 10000)
    ax.plot(x, 0.5*cauchy.pdf(x, loc=3, scale=1), 'r-', lw=3, alpha=0.6, label='loc = 3')
    ax.plot(x, 0.5*cauchy.pdf(x, loc=5, scale=1), 'b-', lw=3, alpha=0.6, label='loc = 5')
    plt.legend(loc='best')
    plt.show()
