from scipy.stats import multivariate_normal


def compute_error_percentage(mu1, cov1, mu2, cov2, x, y):
    z1 = multivariate_normal.pdf([x, y], mean=mu1, cov=cov1)
    z2 = multivariate_normal.pdf([x, y], mean=mu2, cov=cov2)
    return z1, z2


if __name__ == "__main__":
    red_points = [[1.5, 0], [0, 0.5], [2, 1],
                  [1, 1], [0.5, 2], [1.5, 2],
                  [2.5, 2], [1, 3], [2, 3]]

    black_points = [[-2, -1], [-1, -1], [1, -1],
                    [0.5, -0.5], [1.5, -0.5], [-1.5, 0],
                    [-0.5, 0.5], [0.5, 0.5], [1.5, 0.5],
                    [-1.5, 1]]
    compute_error_percentage([1.33, 1.61], [[0.6250125, 0.2083375], [0.2083375, 1.1111125]],
                             [-0.15, -0.15], [[1.725, 0.00277778], [0.00277778, 0.55833333]],
                             1.5, 0)

    mu1 = [1.33, 1.61]
    cov1 = [[0.6250125, 0.2083375], [0.2083375, 1.1111125]]
    mu2 = [-0.15, -0.15]
    cov2 = [[1.725, 0.00277778], [0.00277778, 0.55833333]]

    red_nums = 0
    black_nums = 0

    for point in red_points:
        z1, z2 = compute_error_percentage(mu1, cov1, mu2, cov2, point[0], point[1])
        if z1 > z2:
            red_nums += 1

    for point in black_points:
        z1, z2 = compute_error_percentage(mu1, cov1, mu2, cov2, point[0], point[1])
        if z1 < z2:
            black_nums += 1

    print('percentage of misclassified:', (len(red_points)-red_nums+len(black_points)-black_nums) /
          (len(red_points)+len(black_points)))

