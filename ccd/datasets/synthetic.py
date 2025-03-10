"""
Generation of synthetic datasets.
"""

import numpy as np


def generate_synthetic_data(p, n, d, g):
    """
    Generates the whole dataset by generating n observations. Firstly it generates the sample from
    binary class variable Y, which is a sample of probability p, taken from Bernoulli distribution,
    which is Binomial distribution with only 1 number of tries. Then generates the
    feature vector X, which for X|Y=0 is taken from normal distribution with mean (0,...,0)
    and covariance matrix specified below, and for X|Y=1 is taken from
    normal distribution with mean (1, 1/2, ..., 1/d), and the same covariance matrix,
    which is defined such that in i-th row and j-th column: a[i][j] = g^{|i-j|}.

    :param p: probability of '1'
    :param n: number of observations
    :param d: number of features
    :param g: parameter on which the covariance matrix depends
    :return: the whole dataset
    """
    X = np.zeros((n, d))
    y = np.zeros(n)

    mean_0 = np.zeros(d)
    mean_1 = np.array([1 / i for i in range(1, d+1)])
    cov = np.array([[g ** abs(i - j) for i in range(d)] for j in range(d)])

    for i in range(n):
        y[i] = np.random.binomial(n=1, p=p)
        if y[i] == 0:
            X[i, :] = np.random.multivariate_normal(mean_0, cov)
        if y[i] == 1:
            X[i, :] = np.random.multivariate_normal(mean_1, cov)

    return X, y
