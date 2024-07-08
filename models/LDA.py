import numpy as np
import scipy as sp

from utils.numpy_utils import vcol

class LDA:
    def __init__(self, m):
        self.m = m

    def fit(self, data, label):
        total_mean = data.mean(1)
        within_class_covariance = np.zeros((data.shape[0], data.shape[0]))
        between_class_covariance = np.zeros((data.shape[0], data.shape[0]))

        for i in np.unique(label):
            class_data = data[:, label == i]
            class_mean = class_data.mean(1)
            within_class_covariance += np.dot(class_data - vcol(class_mean), (class_data - vcol(class_mean)).T)
            between_class_covariance += class_data.shape[1] * np.dot(vcol(class_mean - total_mean), vcol(class_mean - total_mean).T)

        within_class_covariance /= data.shape[1]
        between_class_covariance /= data.shape[1]

        eigenvalues, eigenvectors = sp.linalg.eigh(between_class_covariance, within_class_covariance)
        orthogonal_eigenvectors, _, _ = np.linalg.svd(eigenvectors[:, ::-1][:, 0:self.m])


        self.principal_components = orthogonal_eigenvectors[:, 0:self.m]
        transformed_data = self.transform(data)
        mean_0 = np.mean(transformed_data[:, label == 0])
        mean_1 = np.mean(transformed_data[:, label == 1])
        self.threshold = (mean_0 + mean_1) / 2
        self.change = mean_0 > mean_1
        if self.change:
            self.principal_components = -self.principal_components

        return transformed_data, self.principal_components

    def transform(self, data):
        return np.dot(self.principal_components.T, data)

    def score(self, data):
        return self.transform(data)

    def predict_binary(self, data, prior_true=0.5, cost_fp=1, cost_fn=1):
        return self.score(data) > self.threshold
