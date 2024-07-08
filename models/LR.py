import numpy as np
import scipy as sp

class LR:
    def __init__(self, l=0.001, regularization=True):
        self.hyper_params = {
            'l': l
        }
        self.regularization = regularization

    def __obj_function__(self, params, data, label, prior_true=None):
        l = self.hyper_params['l']
        w = params[:-1]
        b = params[-1]

        prior_wheight = np.where(label == 1, prior_true / np.sum(label == 1), (1 - prior_true) / np.sum(label == 0)) if prior_true is not None else np.ones(data.shape[1]) / data.shape[1]

        regularization = l * np.dot(w, w) / 2 if self.regularization else 0
        z = np.where(label == 1, 1, -1)
        function = (np.logaddexp(0, -z * (np.dot(w, data) + b)) * prior_wheight).sum() + regularization
       
        g = -z / (1 + np.exp(z * (np.dot(w, data) + b)))
        gradient_w = np.dot(data, g * prior_wheight)  + l * w
        gradient_b = (g * prior_wheight).sum()
        return function, np.concatenate([gradient_w, [gradient_b]])

    def fit(self, data, label, prior_true=None):
        params = np.zeros(data.shape[0] + 1)

        min_params, min_function, _ = sp.optimize.fmin_l_bfgs_b(self.__obj_function__, params, args=(data, label, prior_true))

        self.w = min_params[:-1]
        self.b = min_params[-1]
        self.empirical_prior_true = np.mean(label) if prior_true is None else prior_true

    def score(self, data):
        log_posterior_ratio = np.dot(self.w, data) + self.b
        log_likelihood_ratio = log_posterior_ratio + np.log(1 - self.empirical_prior_true) - np.log(self.empirical_prior_true)

        return log_likelihood_ratio

    def predict_binary(self, data, prior_true=0.5, cost_fp=1, cost_fn=1):
        score = self.score(data)
        threshold = np.log(((1 - prior_true) * cost_fp)) - np.log((prior_true * cost_fn))

        return score > threshold