"""
Generation of synthetic datasets.
"""

import numpy as np


def generate_y(p, n):
    """
    Generates a class variable y, which is a sample of size n and probability p,
    taken from Bernoulli distribution (which is Binomial distribution with 1 number of tries).
    """
    return np.random.binomial(n=1, p=p, size=n)


def covariance_matrix(d, g):
    """
    As the covariance matrix of both X|Y=0 and X|Y=1 are the same, here we define it as it
    is intended, therefore we have a[i][j] = g^{|i-j|}.
    """
    return np.array([[g ** abs(i - j) for i in range(d)] for j in range(d)])


def generate_X(y, n, d, g):
    """
    Generates the feature vector X, which for X|Y=0 is taken from Normal distribution with mean (0,...,0)
    and covariance as intended and for X|Y=1 is taken from Normal distribution with mean (1, 1/2, ..., 1/d),
    and the same covariance matrix.
    """
    X = np.zeros((n, d))

    mean_0 = np.zeros(d)
    mean_1 = np.array([1 / i for i in range(1, d+1)])

    cov = covariance_matrix(d, g)

    for i in range(n):
        if y[i] == 0:
            X[i, :] = np.random.multivariate_normal(mean_0, cov)
        if y[i] == 1:
            X[i, :] = np.random.multivariate_normal(mean_1, cov)

    return X


def generate_synthetic(p, n, d, g):
    """
    Generates the whole dataset, first by generating the y binary class variable,
    then by creating the data X, following previous assumptions.
    """
    y_vector = generate_y(p, n)
    X_matrix = generate_X(y_vector, n, d, g)

    return X_matrix, y_vector
