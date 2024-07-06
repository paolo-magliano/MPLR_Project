import numpy as np
import scipy as sp

from utils.numpy_utils import vcol
from utils.data import load_data

def PCA(data, m):
    covariance = np.cov(data - vcol(data.mean(1)), bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    principal_components = eigenvectors[:, ::-1][:, 0:m]
    projected_data = np.dot(principal_components.T, data)

    return projected_data, principal_components 

def LDA(data, label, m):
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
    
    principal_components = eigenvectors[:, ::-1][:, 0:m]
    projected_data = np.dot(principal_components.T, data)

    return projected_data, principal_components

