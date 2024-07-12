from utils.data import load_data, shuffle_data, k_data, k_data_calibration, split_db_2to1
from utils.numpy_utils import vcol, vrow
from models.PCA import PCA
from models.LDA import LDA
from models.MVG import MVG, TiedMVG, NaiveMVG, TiedNaiveMVG
from models.LR import LR, QuadraticLR
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
        # data_train, label_train = data_train[:, ::50], label_train[::50]

        # data_train_mean = data_train.mean(axis=1).reshape(-1, 1)
        # data_train = data_train - data_train_mean
        # data_test = data_test - data_train_mean

        if dr is not None:
            dr.fit(data_train)
            data_train = dr.transform(data_train)
            # plt.hist(data_train, label_train, show=True)
            data_test = dr.transform(data_test)

        model.fit(data_train, label_train) # if model.__class__.__name__ != 'LR' else model.fit(data_train, label_train, prior_true)
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
            plt.bayes_error(label_test, model.score(data_test), 2.5, cost_fp, cost_fn)

    error /= k
    DCF /= k
    min_DCF /= k

    # plt.confusion_matrix(label_test, predicted_label)

    print(f'Error rate: {round(error*10000)/100} %')
    print(f'DCF: {round(DCF*1000)/1000}') if DCF != np.inf else print(f'DCF: {DCF}')
    print(f'Min DCF: {round(min_DCF*1000)/1000}') if min_DCF != np.inf else print(f'Min DCF: {min_DCF}')

    return error, DCF, min_DCF

if __name__ == "__main__":

    data, label = load_data("data/trainData.txt")
    data, label = shuffle_data(data, label)

    DRs = [None] # + [PCA(i) for i in range(1, data.shape[0] + 1)]

    # lambdas = numpy.logspace(-5, -1, 21)
    Cs = numpy.logspace(-3, 0, 7)
    sigmas = [1e-4, 1e-3, 1e-2, 1e-1]

    k = 3
    prior_true, cost_fp, cost_fn = 0.1, 1, 1
    0# models = [RBFSVM(C=C, sigma=sigma) for C in Cs] 
    plot = False
    global_errors, global_DCFs, global_min_DCFs = [], [], []

    for dr in DRs:
        for sigma in sigmas:
            models = [RBFSVM(C=C, sigma=sigma, K=1) for C in Cs]
            errors, DCFs, min_DCFs = [], [], [] 
            for model in models:
                print(f'Model: {model.__class__.__name__} DR: {dr.__class__.__name__+ " " + str(dr.m) if dr is not None else "None"}')
                if model.__class__.__name__ == 'LR':
                    print(f'Lambda: {model.hyper_params["l"]}')
                if model.__class__.__name__ == 'SVM' or model.__class__.__name__ == 'PoliSVM' or model.__class__.__name__ == 'RBFSVM':
                    print(f'C: {model.hyper_params["C"]}')
                    if model.__class__.__name__ == 'RBFSVM':
                        print(f'Sigma: {model.hyper_params["sigma"]}')
                error, DCF, min_DCF = k_fold(data, label, model, k, prior_true, cost_fp, cost_fn, dr, plot)
                errors.append(error)
                DCFs.append(DCF)
                min_DCFs.append(min_DCF)
            global_errors.append(np.array(errors))
            global_DCFs.append(np.array(DCFs))
            global_min_DCFs.append(np.array(min_DCFs))
    print(f'Shape: {np.array(global_DCFs).shape}')
    plt.hyper_params(Cs, np.array(global_DCFs), np.array(global_min_DCFs), name='C', lines=sigmas)

    params = np.array([(C, sigma) for C in Cs for sigma in sigmas])
    
    print(f'Best model params (C, sigma): {params[np.argmin(np.array(global_min_DCFs))]} with Min DCF: {np.min(np.array(global_min_DCFs))}')

