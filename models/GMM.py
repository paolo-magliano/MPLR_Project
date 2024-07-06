import numpy as np
import scipy as sp
import json
import os

from models.MVG import logpdf_GAU_ND
from utils.numpy_utils import vrow, vcol

def logpdf_GMM(data, means, covariances, weights):
    log_likelihood = np.array([logpdf_GAU_ND(data, mean, covariance) for mean, covariance in zip(means, covariances)])
    log_joint = log_likelihood + np.log(weights)
    log_marginal = sp.special.logsumexp(log_joint, axis=0)
    return log_marginal

class GMM:
    def __init__(self, m=2):
        self.hyper_params = {
            'm': m
        }

    def __constraint__(self, covariance):
        U, S, _ = np.linalg.svd(covariance)
        S[S < 1e-2] = 1e-2
        return np.dot(U, np.dot(np.diag(S), U.T))

    def __e_step__(self, data, means, covariances, weights):
        log_likelihood = np.array([logpdf_GAU_ND(data, mean, covariance) for mean, covariance in zip(means, covariances)])
        log_joint = log_likelihood + np.log(weights)
        log_marginal = sp.special.logsumexp(log_joint, axis=0)
        posterior = np.exp(log_joint - log_marginal)

        return posterior

    def __m_step__(self, data, posterior):
        z = vcol(posterior.sum(1))
        f = np.dot(posterior, data.T)
        e = np.einsum('ij, kj -> ikj', data, data)
        s = np.einsum('ij, xzj -> ixz', posterior, e)

        means = f / z
        covariances = s / z[..., np.newaxis] - means[..., np.newaxis] * means[:, np.newaxis, :]
        weights = z / data.shape[1]

        covariances = np.array([self.__constraint__(covariance) for covariance in covariances])

        return means[..., np.newaxis], covariances, weights

    def __init_params__(self, data, means=None, covariances=None, weights=None):
        if means is None or covariances is None or weights is None:
            new_means = data.mean(1)[np.newaxis, :, np.newaxis]
            new_covariances = np.cov(data, bias=True).reshape(1, data.shape[0], data.shape[0])
            new_weights = np.ones((1, 1))

            new_covariances = np.array([self.__constraint__(covariance) for covariance in new_covariances])
        else:
            new_weights = np.concatenate([[weight / 2, weight / 2] for weight in weights])
            USV = [np.linalg.svd(covariance) for covariance in covariances]
            new_means = np.concatenate([[mean - (U[:, 0:1] * S[0]**0.5 * 0.1), mean + (U[:, 0:1] * S[0]**0.5 * 0.1)] for mean, (U, S, V) in zip(means, USV)])
            new_covariances = np.concatenate([[covariance, covariance] for covariance in covariances])

        return new_means, new_covariances, new_weights

    def __em__step__(self, data, means, covariances, weights):
        threshold = 1e-6
        means, covariances, weights = self.__init_params__(data, means, covariances, weights)

        log_likelihood_old = -np.inf
        log_likelihood_new = logpdf_GMM(data, means, covariances, weights).mean()

        while log_likelihood_new - log_likelihood_old > threshold:
            posteriors = self.__e_step__(data, means, covariances, weights)
            means, covariances, weights = self.__m_step__(data, posteriors)

            log_likelihood_old = log_likelihood_new
            log_likelihood_new = logpdf_GMM(data, means, covariances, weights).mean()

        print(f'Log likelihood: {log_likelihood_new}')

        return means, covariances, weights

    def fit(self, data, label):
        means = [None for _ in np.unique(label)]
        covariances = [None for _ in np.unique(label)]
        weights = [None for _ in np.unique(label)]
        for i in np.unique(label):
            while means[i] is None or means[i].shape[0] < self.hyper_params['m']:
                print(f'Fitting GMM with {1 if means[i] is None else means[i].shape[0] * 2} components for class {i}')
                means[i], covariances[i], weights[i] = self.__em__step__(data[:, label == i], means[i], covariances[i], weights[i])

        self.means, self.covariances, self.weights = np.array(means), np.array(covariances), np.array(weights)

    def score(self, data):
        log_likelihood_true = logpdf_GMM(data, self.means[1], self.covariances[1], self.weights[1])
        log_likelihood_false = logpdf_GMM(data, self.means[0], self.covariances[0], self.weights[0])
        log_likelihood_ratio = log_likelihood_true - log_likelihood_false

        return log_likelihood_ratio

    def predict_binary(self, data, prior_true=0.5, cost_fp=1, cost_fn=1):
        score = self.score(data)
        threshold = np.log(((1 - prior_true) * cost_fp)) - np.log((prior_true * cost_fn))

        return score > threshold

    def predict(self, data, prior):
        log_likelihood = np.array([logpdf_GMM(data, self.means[i], self.covariances[i], self.weights[i]) for i in range(self.means.shape[0])])
        log_joint = log_likelihood + np.log(prior)
        log_marginal = sp.special.logsumexp(log_joint, axis=0)
        posterior = np.exp(log_joint - log_marginal)

        return np.argmax(posterior, axis=0)

class DiagonalGMM(GMM):
    def __m_step__(self, data, posterior):
        z = vcol(posterior.sum(1))
        f = np.dot(posterior, data.T)
        e = np.einsum('ij, kj -> ikj', data, data)
        s = np.einsum('ij, xzj -> ixz', posterior, e)

        means = f / z
        covariances = s / z[..., np.newaxis] - means[..., np.newaxis] * means[:, np.newaxis, :]
        covariances = np.array([np.diag(np.diag(covariance)) for covariance in covariances])
        weights = z / data.shape[1]

        covariances = np.array([self.__constraint__(covariance) for covariance in covariances])

        return means[..., np.newaxis], covariances, weights

    def __init_params__(self, data, means=None, covariances=None, weights=None):
        if means is None or covariances is None or weights is None:
            new_means = data.mean(1)[np.newaxis, :, np.newaxis]
            new_covariances = np.diag(np.diag(np.cov(data, bias=True))).reshape(1, data.shape[0], data.shape[0])
            new_weights = np.ones((1, 1))

            new_covariances = np.array([self.__constraint__(covariance) for covariance in new_covariances])
        else:
            new_means, new_covariances, new_weights = super().__init_params__(data, means, covariances, weights)

        return new_means, new_covariances, new_weights

class TiedGMM(GMM):
    def __m_step__(self, data, posterior):
        z = vcol(posterior.sum(1))
        f = np.dot(posterior, data.T)
        e = np.einsum('ij, kj -> ikj', data, data)
        s = np.einsum('ij, xzj -> ixz', posterior, e)

        means = f / z
        covariances = s / z[..., np.newaxis] - means[..., np.newaxis] * means[:, np.newaxis, :]
        weights = z / data.shape[1]
        covariances = np.array([weight * covariance for weight, covariance in zip(weights, covariances)])

        covariances = np.array([self.__constraint__(covariance) for covariance in covariances])

        return means[..., np.newaxis], covariances, weights

