"""
Implementation of the regularized logistic regression using CCD algorithm.
"""

import matplotlib.pyplot as plt
import numpy as np


class LogRegCCD:
    def __init__(self):
        self.lambda_: float = None
        self.coefficients: np.ndarray = None
        self.intercept: float = None
        self.x_train: np.ndarray = None
        self.y_train: np.ndarray = None
        self.x_valid: np.ndarray = None
        self.y_valid: np.ndarray = None

    def fit(self, x_train, y_train):
        pass

    def validate(self, x_valid, y_valid, measure):
        pass

    def predict_proba(self):
        pass

    def plot(self, measure, lambda_range=None, num_points=10):
        """
        Produces a plot showing how the given evaluation measure changes with the lambda parameter.
        Available measures: 'recall', 'precision', 'f1', 'balanced_accuracy', 'roc_auc'
        """
        if lambda_range is None:
            lambda_range = np.logspace(-4, 2, num_points)

        measure_values = []

        for i in lambda_range:
            self.lambda_param = i
            self.fit(self.x_train, self.y_train)
            current_measure = self.validate(self.x_valid, self.y_valid, measure)
            measure_values.append(current_measure)
        plt.figure()
        plt.plot(lambda_range, measure_values, marker="x")
        plt.xlabel("Lambda (Regularization Strength)")
        plt.ylabel(measure)
        plt.title(f"Validation {measure} vs Lambda")
        plt.grid(True)
        plt.show()

    def plot_coefficients(self, lambda_range=None, num_points=10):
        """
        Produces a plot showing the coefficients values as function of the lambda parameter.
        """
        if lambda_range is None:
            lambda_range = np.logspace(-4, 2, num_points)

        coeffs_list = []

        for i in lambda_range:
            self.lambda_param = i
            self.fit(self.X_train, self.y_train)
            coeffs_list.append(self.coefficients.copy())

        # Expected shape: (num_points, number_of_features)
        coeffs = np.array(coeffs_list)

        plt.figure()
        num_coeffs = coeffs.shape[1]
        for i in range(num_coeffs):
            plt.plot(lambda_range, coeffs[:, i], marker="x", label=f"Coefficient {i}")
        plt.xlabel("Lambda (Regularization Strength)")
        plt.ylabel("Coefficient Value")
        plt.title("Coefficients vs Lambda")
        plt.legend()
        plt.grid(True)
        plt.show()
