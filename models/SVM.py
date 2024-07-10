import numpy as np
import scipy as sp

from utils.numpy_utils import vcol

class SVM:
    def __init__(self, K=1, C=1):
        self.hyper_params = {
            'K': K,
            'C': C
        }

    def __kernel__(self, data1, data2):
        return np.dot(data1.T, data2)

    def __regularization__(self, data):
        return np.vstack([data, np.ones(data.shape[1]) * self.hyper_params['K']])

    def __obj_function__(self, params, data, label):
        alpha = params

        z = np.where(label == 1, 1, -1)
        H = self.__kernel__(data, data) * np.outer(z, z)
        function = np.dot(alpha, np.dot(H, alpha)) / 2 - np.sum(alpha) 

        gradient = np.dot(H, alpha) - 1

        return function, gradient

    def fit(self, data, label):
        params = np.zeros(data.shape[1])
        bounds = [(0, self.hyper_params['C']) for _ in range(data.shape[1])]
        data = self.__regularization__(data)

        min_params, min_function, info = sp.optimize.fmin_l_bfgs_b(self.__obj_function__, params, args=(data, label), bounds=bounds, factr=1.0)

        self.alpha = min_params
        self.data_train = data
        self.z = np.where(label == 1, 1, -1)
  
    def score(self, data):
        data = self.__regularization__(data)

        return np.dot(self.alpha * self.z, self.__kernel__(self.data_train, data))

    def predict_binary(self, data, prior_true=0.5, cost_fp=1, cost_fn=1):
        return self.score(data) > 0

class PoliSVM(SVM):
    def __init__(self, K=0, C=1, d=2, c=1):
        super().__init__(K, C)
        self.hyper_params['d'] = d
        self.hyper_params['c'] = c

    def __kernel__(self, data1, data2):
        return (np.dot(data1.T, data2) + self.hyper_params['c']) ** self.hyper_params['d'] + self.hyper_params['K'] ** 2

    def __regularization__(self, data):
        return data   

    def score(self, data):
        data = self.__regularization__(data)
        score = np.dot(self.alpha * self.z, self.__kernel__(self.data_train, data))

        return score

class RBFSVM(SVM):
    def __init__(self, K=0, C=1, sigma=1):
        super().__init__(K, C)
        self.hyper_params['sigma'] = sigma

    def __kernel__(self, data1, data2):
        sq_distance = vcol(np.sum(data1**2, axis=0)) + np.sum(data2**2, axis=0) - 2 * np.dot(data1.T, data2)
        return np.exp(-self.hyper_params['sigma'] * sq_distance) + self.hyper_params['K'] ** 2


    def __regularization__(self, data):
        return data

    def score(self, data):
        data = self.__regularization__(data)
        score = np.dot(self.alpha * self.z, self.__kernel__(self.data_train, data))

        return score


