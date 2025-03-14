"""
Getting real datasets.
The first dataset, which is a dataset "spambase" is taken from OpenML. The target variable y checked whether
the mail was a spam, based on numerical metrics from the text of a mail.
"""

import numpy as np
import pandas as pd
from scipy.io import arff

np.random.seed(42)


def load_data(file):
    """
    Loads the file with .arff format into pandas dataframe.
    :param file: the .arff file
    :return: pandas dataframe
    """
    data, meta = arff.loadarff(file)
    dataframe = pd.DataFrame(data)
    return dataframe


def fill_dummy(df, parameter=0.6):
    """
    It creates a list of the permuted (by columns) dataframes, and then it concatenates them,
    such that the number of observations will be sufficient to the project requirements.
    :param df: inserted pandas dataframe
    :param parameter: by default 0.6, the project insists on a number bigger than 0.5
    :return: dataframe with necessary dimensions
    """
    number_of_copies = int(df.shape[0] / df.shape[1] * parameter) + 1
    permuted_dfs = [df]
    for _ in range(number_of_copies):
        permutation = df.apply(np.random.permutation)
        permuted_dfs.append(permutation)
    new_df = pd.concat(permuted_dfs, axis=1)
    return new_df


def missing_values_check(df):
    """
    Checks whether a dataframe contains missing values.
    :param df: pandas dataframe
    :return: true if the dataframe is filled only with numbers, as it is needed to be
    """
    missing_values = pd.isna(df).any().any()
    return not missing_values


def collinearity_check(df):
    """
    Checks whether a dataframe have pairwise linearly independent variables (no collinearity).
    :param df: pandas dataframe
    :return: true if the dataframe contains pairwise linearly independent variables
    """
    rank = np.linalg.matrix_rank(df)
    fewer_dimension = min(df.shape[0], df.shape[1])
    return rank == fewer_dimension


def get_dataset_1():
    """
    First it loads the first dataset into pandas dataframe, then creates the target variable y.
    Then, it drops the column with target variable from original dataframe and enlarges to meet the
    project requirements. Lastly, it prints out whether the necessary condition hold. Both need
    to be true in order to work on this dataset later on.
    :return: cleaned input features X and target variable y
    """
    df1 = load_data("dataset_1.arff")

    y = df1["class"].astype(int)
    y = y.to_numpy()

    X_original = df1.drop(columns=["class"], axis=1)
    X = fill_dummy(X_original)
    print(f"Does the first dataset have all filled values? {missing_values_check(X)}.")
    print(
        f"Does the first dataset have no collinear variables? {collinearity_check(X)}."
    )
    X = X.to_numpy()

    return X, y


get_dataset_1()
