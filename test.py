from utils.data import load_data, shuffle_data, k_data, k_data_calibration
from utils.numpy_utils import vcol, vrow
from models.PCA import PCA
from models.LDA import LDA
from models.MVG import MVG, TiedMVG, NaiveMVG, TiedNaiveMVG
from models.LR import LR
from models.SVM import SVM, PoliSVM, RBFSVM
from models.GMM import GMM, DiagonalGMM, TiedGMM
from utils.evaluation import *
import utils.plot as plt

import numpy
import tqdm

def k_fold(data, label, model, k, prior_true, cost_fp, cost_fn, dr=None, plot=False, leave_one_out=False):
    error = 0
    DCF = 0
    min_DCF = 0

    if leave_one_out:
        k = data.shape[1]

    for i in tqdm.tqdm(range(k), desc='K-Fold'):
        (data_train, label_train), (data_test, label_test) = k_data(data, label, i, k)

        if dr is not None:
            dr.fit(data_train)
            data_train = dr.transform(data_train)
            # plt.hist(data_train, label_train, show=True)
            data_test = dr.transform(data_test)

        model.fit(data_train, label_train)
        # plt.hist(model.transform(data_train), label_train, show=True)
        predicted_label = model.predict_binary(data_test, prior_true, cost_fp, cost_fn)

        error += error_rate(label_test, predicted_label)
        DCF += normalized_DCF_binary(label_test, predicted_label, prior_true, cost_fp, cost_fn)
        model_min_DCF, _ = min_normalized_DCF_binary(label_test, model.score(data_test), prior_true, cost_fp, cost_fn)
        min_DCF += model_min_DCF

        # Plot
        if plot:
            if model.__class__.__name__ == 'MVG' or model.__class__.__name__ == 'NaiveMVG' or model.__class__.__name__ == 'TiedMVG':
                plt.gaussian_hist(data_train, label_train, model.mean, model.covariance)
            elif model.__class__.__name__ == 'GMM' or model.__class__.__name__ == 'DiagonalGMM' or model.__class__.__name__ == 'TiedGMM':
                plt.gmm_hist(data_train, label_train, model.means, model.covariances, model.weights)
            # plt.ROC_curve(label_test, model.score(data_test))
            # plt.bayes_error(label_test, model.score(data_test), 2.5, cost_fp, cost_fn)

    error /= k
    DCF /= k
    min_DCF /= k

    print(f'Error rate: {round(error*10000)/100} %')
    print(f'DCF: {round(DCF*1000)/1000}')
    print(f'Min DCF: {round(min_DCF*1000)/1000}')

if __name__ == "__main__":
    data, label = load_data("data/trainData.txt")
    data, label = shuffle_data(data, label)

    DRs = [PCA(i) for i in range(1, data.shape[0] + 1)]

    k = 10
    prior_true, cost_fp, cost_fn = 0.5, 1, 1
    models = [MVG(), TiedMVG(), NaiveMVG()]
    plot = False

    for dr in DRs:
        for model in models:
            print(f'Model: {model.__class__.__name__} DR: {dr.__class__.__name__+ " " + str(dr.m) if dr is not None else "None"}')
            k_fold(data, label, model, k, prior_true, cost_fp, cost_fn, dr, plot)







