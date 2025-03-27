"""
Implementation of the regularized logistic regression using CCD algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    average_precision_score,
)

matplotlib.use("TkAgg")


def sigmoid(t):
    """
    Computes the sigmoid function.

    :param t: numpy array
    :return: sigmoid-transformed numpy array
    """
    return 1 / (1 + np.exp(-t))


def standardize(X):
    """
    Standardizes the dataset by subtracting the mean and dividing by the standard deviation.

    :param X: dataset X
    :return: standardized X
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std


def soft_thresholding(z, gamma):
    """
    Applies the soft-thresholding operator for Lasso regularization.

    :param z: input value
    :param gamma: thresholding parameter
    :return: soft-threshold value
    """
    if gamma < abs(z):
        if z > 0:
            return z - gamma
        else:
            return z + gamma
    else:
        return 0


class LogRegCCD:
    """
    Regularized Logistic Regression using the Cyclic Coordinate Descent (CCD) algorithm.
    Supports L1 (Lasso), L2 (Ridge), and ElasticNet regularization.
    """

    def __init__(
        self,
        alpha=1,
        lambda_max=1 / 10,
        lambda_min=1 / 10**5,
        number_of_iterations=20,
        middle_loop_iterations=50,
        threshold=1 / 10**5,
        epsilon=1 / 10**5,
    ):
        # change alpha to an arbitrary value from (0,1) for ElasticNet
        """
        Initializes the logistic regression model with regularization.

        :param alpha: ElasticNet mixing parameter (1 = Lasso, 0 = Ridge, anything from (0,1) = ElasticNet)
        :param lambda_max: maximum lambda value
        :param lambda_min: minimum lambda value
        :param number_of_iterations: number of lambda values
        :param middle_loop_iterations: number of middle loop iterations
        :param threshold: convergence threshold for coordinate descent
        :param epsilon: epsilon for numerical stability
        """
        self.alpha = alpha
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.number_of_iterations = number_of_iterations
        self.middle_loop_iterations = middle_loop_iterations
        self.threshold = threshold
        self.epsilon = epsilon
        self.lambda_values = np.logspace(
            np.log10(self.lambda_max),
            np.log10(self.lambda_min),
            self.number_of_iterations,
        )  # ZAPYTAĆ
        self.beta0_path = []
        self.beta_path = []
        self.best_lambda = None
        self.best_beta0 = None
        self.best_beta = None

    def fit(self, X_train, y_train):
        """
        Fits the logistic regression model using the CCD algorithm.

        :param X_train: training feature matrix
        :param y_train: training target vector
        """
        n_observations, p_features = X_train.shape

        beta0 = 0
        beta = np.zeros(p_features)  # ZAPYTAĆ

        for each_lambda in self.lambda_values:
            for _ in range(self.middle_loop_iterations):
                beta0_old = beta0
                beta_old = beta.copy()

                p = sigmoid(beta0 + X_train @ beta)
                p[p < self.epsilon] = 0  # ZAPYTAĆ
                p[p > 1 - self.epsilon] = 1

                w = p * (1 - p)
                w[p < self.epsilon] = self.epsilon
                w[p > 1 - self.epsilon] = self.epsilon

                z = beta0 + X_train @ beta + (y_train - p) / w

                beta0 = np.mean(z - X_train @ beta)

                for j in range(p_features):
                    z_j = beta0 + X_train @ beta - X_train[:, j] * beta[j]
                    denominator = np.mean(w * X_train[:, j] ** 2) + each_lambda * (
                        1 - self.alpha
                    )
                    nominator = soft_thresholding(
                        np.mean(w * X_train[:, j] * (z - z_j)), each_lambda * self.alpha
                    )
                    beta[j] = nominator / denominator

                params_old = np.concatenate(([beta0_old], beta_old))
                params_new = np.concatenate(([beta0], beta))
                params_diff = np.linalg.norm(params_new - params_old)
                if params_diff < self.threshold:
                    break

            self.beta0_path.append(beta0)
            self.beta_path.append(beta.copy())

        self.beta0_path = np.array(self.beta0_path)
        self.beta_path = np.array(self.beta_path)

    def validate(self, X_valid, y_valid, measure):
        """
        Validates the model and selects the best lambda based on a given evaluation measure.

        :param X_valid: validation feature matrix
        :param y_valid: validation target vector
        :param measure: selected evaluation measure:
        ('recall', 'precision', 'F-measure', 'balanced accuracy', 'ROC AUC', 'AUPRC')
        :return: best score
        """
        best_score = -np.inf

        for i, each_lambda in enumerate(self.lambda_values):
            beta0 = self.beta0_path[i]
            beta = self.beta_path[i]

            probabilities = sigmoid(beta0 + X_valid @ beta)
            predictions = (probabilities >= 0.5).astype(int)

            if measure == "recall":
                score = recall_score(y_valid, predictions)
            elif measure == "precision":
                score = precision_score(y_valid, predictions)
            elif measure == "F-measure":
                score = f1_score(y_valid, predictions)
            elif measure == "balanced accuracy":
                score = balanced_accuracy_score(y_valid, predictions)
            elif measure == "ROC AUC":
                score = roc_auc_score(y_valid, probabilities)
            elif measure == "AUPRC":
                score = average_precision_score(y_valid, probabilities)
            else:
                raise ValueError("Invalid measure selected.")

            if score > best_score:
                best_score = score
                self.best_lambda = each_lambda
                self.best_beta0 = beta0
                self.best_beta = beta
        return best_score

    def predict_proba(self, X_test):
        """
        Predicts probabilities for test data.

        :param X_test: test feature matrix
        :return: predicted probabilities
        """
        probabilities = sigmoid(self.best_beta0 + X_test @ self.best_beta)
        return probabilities

    def plot(self, measure, X_valid, y_valid):
        """
        Produces a plot showing how the given evaluation measure changes with the lambda parameter.

        :param measure: The evaluation metric to be used. Options include 'recall', 'precision', 'F-measure',
                        'balanced accuracy', 'ROC AUC', and 'AUPRC'.
        :param X_valid: The validation set features as a NumPy array of shape (n_samples, n_features).
        :param y_valid: The true labels of the validation set as a NumPy array of shape (n_samples,).
        """
        scores = []

        for i, each_lambda in enumerate(self.lambda_values):
            beta0 = self.beta0_path[i]
            beta = self.beta_path[i]

            probabilities = sigmoid(beta0 + X_valid @ beta)
            predictions = (probabilities >= 0.5).astype(int)

            if measure == "recall":
                score = recall_score(y_valid, predictions)
            elif measure == "precision":
                score = precision_score(y_valid, predictions)
            elif measure == "F-measure":
                score = f1_score(y_valid, predictions)
            elif measure == "balanced accuracy":
                score = balanced_accuracy_score(y_valid, predictions)
            elif measure == "ROC AUC":
                score = roc_auc_score(y_valid, probabilities)
            elif measure == "AUPRC":
                score = average_precision_score(y_valid, probabilities)
            else:
                raise ValueError("Invalid measure selected.")

            scores.append(score)

        plt.figure(figsize=(10, 6))
        plt.plot(self.lambda_values, scores, marker="o")
        plt.xscale("log")
        plt.xlabel("lambda")
        plt.ylabel(measure)
        plt.title(f"{measure} depending on the lambda")
        plt.grid(True)
        plt.show()

    def plot_coefficients(self):
        """
        Produces a plot of the coefficient values against different values of the lambda parameter.
        """
        plt.figure(figsize=(10, 6))
        for i in range(np.array(self.beta_path).shape[1]):
            plt.plot(
                self.lambda_values, self.beta_path[:, i], label=f"Coefficient {i + 1}"
            )
        plt.xscale("log")
        plt.xlabel("lambda")
        plt.ylabel("Coefficient Value")
        plt.title("Coefficient Values depending on lambda")
        plt.legend()
        plt.grid(True)
        plt.show()
