"""
Getting real datasets.

The first dataset, which is a dataset "spambase" is taken from OpenML.
It has 58 different variables and 4601 number of observations.
The target variable y checked whether the mail was a spam,
based on numerical metrics from the text of a mail.

The second dataset, which is a dataset "blood-transfusion-service-center" is also taken from OpenML.
It has 5 different variables and 748 number of observations.
The target variable y checked whether the patient donated blood in a specific month,
based on numerical metrics of previous donations.
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


def missing_values_check(array):
    """
    Checks whether an array contains missing values.

    :param array: numpy array
    :return: true if the numpy array is filled only with numbers, as it is needed to be
    """
    missing_values = np.isnan(array).any()
    return not missing_values


def collinearity_check(df):
    """
    Checks whether there is no such two columns, that one is a scalar multiple of another (collinearity).

    :param df: pandas dataframe
    :return: true if there is no collinearity between variables
    """
    correlation_matrix = df.corr()
    np.fill_diagonal(correlation_matrix.values, 0)
    return not (correlation_matrix == 1).any().any()


def get_dataset_1(path="dataset_1.arff"):
    """
    First it loads the first dataset into pandas dataframe, then creates the target variable y.
    Then, it drops the column with target variable from original dataframe and enlarges to meet the
    project requirements.

    :return: cleaned input features X and target variable y
    """
    df1 = load_data(path)

    y = df1["class"].astype(int)
    y = y.to_numpy()

    X_original = df1.drop(columns=["class"], axis=1)
    X = fill_dummy(X_original)
    X = X.to_numpy()

    return X, y


def get_dataset_2(path="dataset_2.arff"):
    """
    First it loads the second dataset into pandas dataframe, then creates the target variable y.
    As the target variable is 1-2 and not 0-1, we subtract one from each value.
    Then, it drops the column with target variable from original dataframe and
    enlarges to meet the project requirements.

    :return: cleaned input features X and target variable y
    """
    df2 = load_data(path)

    y = df2["Class"].astype(int)
    y = y - 1
    y = y.to_numpy()

    X_original = df2.drop(columns=["Class"], axis=1)
    X = fill_dummy(X_original)
    X = X.to_numpy()

    return X, y


if __name__ == "__main__":
    datasets = [get_dataset_1(), get_dataset_2()]

    for i, (X_data, y_data) in enumerate(datasets):
        print(
            f"Does the dataset number {i + 1} have all filled values? "
            f"{missing_values_check(X_data)}."
        )
        print(
            f"Does the dataset number {i + 1} have no collinear variables? "
            f"{collinearity_check(pd.DataFrame(X_data))}."
        )
        print(
            f"Number of observations: {X_data.shape[0]}. Number of variables: {X_data.shape[1]}."
        )
