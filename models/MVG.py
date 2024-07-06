import numpy as np
import scipy as sp

from utils.numpy_utils import vcol

def logpdf_GAU_ND(data, mean, covariance):
    centered_data = data - mean

    constant_term = -0.5 * data.shape[0] * np.log(2 * np.pi)
    log_determinat = np.linalg.slogdet(covariance)[1]

    return constant_term - 0.5 * log_determinat - 0.5 * (centered_data * np.dot(np.linalg.inv(covariance), centered_data)).sum(0)

class MVG:
    def __init__(self):
        pass

    def fit(self, data, label):
        self.mean = np.array([vcol(data[:, label == i].mean(1)) for i in np.unique(label)])
        self.covariance = np.array([np.cov(data[:, label == i], bias=True) for i in np.unique(label)])

    def score(self, data):
        log_likelihood_true = logpdf_GAU_ND(data, self.mean[1], self.covariance[1])
        log_likelihood_false = logpdf_GAU_ND(data, self.mean[0], self.covariance[0])
        log_likelihood_ratio = log_likelihood_true - log_likelihood_false

        return log_likelihood_ratio

    def predict_binary(self, data, prior_true=0.5, cost_fp=1, cost_fn=1):
        score = self.score(data)
        threshold = np.log(((1 - prior_true) * cost_fp)) - np.log((prior_true * cost_fn))

        return score > threshold

    def predict(self, data, prior):
        log_likelihood = np.array([logpdf_GAU_ND(data, self.mean[i], self.covariance[i]) for i in range(self.mean.shape[0])])
        log_joint = log_likelihood + np.log(prior)
        log_marginal = sp.special.logsumexp(log_joint, axis=0)
        posterior = np.exp(log_joint - log_marginal)

        return np.argmax(posterior, axis=0)

class TiedMVG(MVG):
    def __init__(self):
        super().__init__()

    def fit(self, data, label):
        self.mean = np.array([vcol(data[:, label == i].mean(1)) for i in np.unique(label)])
        class_covariance = np.array([np.cov(data[:, label == i], bias=True) for i in np.unique(label)])
        self.covariance = np.array([np.mean(class_covariance, axis=0) for _ in np.unique(label)])

class NaiveMVG(MVG):
    def __init__(self):
        super().__init__()

    def fit(self, data, label):
        self.mean = np.array([vcol(data[:, label == i].mean(1)) for i in np.unique(label)])
        self.covariance = np.array([np.diag(np.diag(np.cov(data[:, label == i], bias=True))) for i in np.unique(label)])

class TiedNaiveMVG(MVG):
    def __init__(self):
        super().__init__()

    def fit(self, data, label):
        self.mean = np.array([vcol(data[:, label == i].mean(1)) for i in np.unique(label)])
        class_covariance = np.array([np.diag(np.diag(np.cov(data[:, label == i], bias=True))) for i in np.unique(label)])
        self.covariance = np.array([np.mean(class_covariance, axis=0) for _ in np.unique(label)])
    