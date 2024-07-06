from utils.dimensionality_reduction import PCA, LDA
from utils.data import load_data, split_db_2to1, split_db, k_fold_data
from utils.numpy_utils import vcol, vrow
from models.MVG import MVG, TiedMVG, NaiveMVG, TiedNaiveMVG
from models.LR import LR
from models.SVM import SVM, PoliSVM, RBFSVM
from models.GMM import GMM, DiagonalGMM, TiedGMM
from utils.evaluation import *
import utils.plot as plt

import numpy
import tqdm

def plot_gaussian_dist():
    data, label = load_data("data/trainData.txt")

    num_labels = len(numpy.unique(label))

    class_means = np.array([data[:, label == i].mean(1) for i in range(num_labels)]).T.reshape(data.shape[0], num_labels, 1, 1)

    class_covariances = np.array([np.cov(data[:, label == i], bias=True) for i in range(num_labels)]).T.reshape(data.shape[0], data.shape[0], num_labels, 1, 1)

    plt.gaussian_hist(data, label, class_means, class_covariances)

def k_fold(k=3, leave_one_out=False):
    data, label = load_data("data/trainData.txt")

    if leave_one_out:
        k = data.shape[1]

    k_data, k_label = split_db(data, label, k)

    accuracy = 0

    for i in tqdm.tqdm(range(k), desc='K-Fold'):
        (data_train, label_train), (data_test, label_test) = k_fold_data(k_data, k_label, i)

        model = NaiveMVG()
        model.fit(data_train, label_train)

        predicted_label = model.predict(data_test, np.ones((np.unique(label_train).shape[0], 1)) / np.unique(label_train).shape[0])

        accuracy += (predicted_label == label_test).sum() / len(label_test)

    accuracy /= k

    print(f'Accuracy: {accuracy*100} - Error rate: {100 - accuracy*100}')

if __name__ == "__main__":
    data, label = load_data("data/iris.txt")

    data = data[:, label != 0]
    label = label[label != 0] 
    label[label == 2] = 0

    # data_gmm = np.load('data/GMM_data_4D.npy')

    (data_train, label_train), (data_test, label_test) = split_db_2to1(data, label)

    prior_true, cost_fp, cost_fn = 0.5, 1, 1

    model = GMM(m=4)
    model.fit(data_train, label_train)

    print(f'Means: {model.means.shape}')
    print(f'Covariances: {model.covariances.shape}')
    print(f'Weights: {model.weights.shape}')

    plt.gmm_hist(data_train, label_train, model.means, model.covariances, model.weights)

    predicted_label = model.predict_binary(data_test, prior_true, cost_fp, cost_fn)

    conf_matrix = confusion_matrix(label_test, predicted_label)

    print(f'Confusion matrix: \n{conf_matrix}')

    error_rate = error_rate(label_test, predicted_label)

    print(f'Error rate: {round(error_rate*10000)/100}')

    DCF = DCF_binary(label_test, predicted_label, prior_true, cost_fp, cost_fn)

    print(f'DCF: {round(DCF*1000)/1000}')

    normalized_DCF = normalized_DCF_binary(label_test, predicted_label, prior_true, cost_fp, cost_fn)

    print(f'Normalized DCF: {round(normalized_DCF*1000)/1000}')

    min_normalized_DCF, threshold = min_normalized_DCF_binary(label_test, model.score(data_test), prior_true, cost_fp, cost_fn)

    print(f'Min normalized DCF: {round(min_normalized_DCF*1000)/1000} - Threshold: {threshold}')

    TPR, FPR, thresholds = ROC_binary(label_test, model.score(data_test))

    plt.ROC_curve(label_test, model.score(data_test))

    plt.bayes_error(label_test, model.score(data_test), 2.5, cost_fp, cost_fn)


