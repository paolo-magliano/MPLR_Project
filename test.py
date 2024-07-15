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

def k_fold(data, label, model, k, prior_true, cost_fp, cost_fn, dr=None, plot=False, calibration=False, leave_one_out=False):
    error = 0
    DCF = 0
    min_DCF = 0

    if calibration:
        error_cal = np.zeros_like(calibrations)
        DCF_cal = np.zeros_like(calibrations)
        min_DCF_cal = np.zeros_like(calibrations)

    if leave_one_out:
        k = data.shape[1]

    for i in tqdm.tqdm(range(k), desc='K-Fold'):
        if calibration:
            (data_train, label_train), (data_test, label_test), (data_cal, label_cal) = k_data_calibration(data, label, i, k)
        else:
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

        if calibration:
                for i, cal_priors in enumerate(calibration): 
                    cal_model = LR(regularization=False)
                    score_train = vrow(model.score(data_cal))
                    score_test = vrow(model.score(data_test))
                    cal_model.fit(score_train, label_cal)
                    predicted_label_cal = cal_model.predict_binary(score_test, prior_true, cost_fp, cost_fn)

                    error_cal[i] += error_rate(label_test, predicted_label_cal)
                    DCF_cal[i] += normalized_DCF_binary(label_test, predicted_label_cal, prior_true, cost_fp, cost_fn)
                    model_min_DCF_cal, _ = min_normalized_DCF_binary(label_test, cal_model.score(score_test), prior_true, cost_fp, cost_fn)
                    min_DCF_cal[i] += model_min_DCF_cal
            

        # Plot
        if plot:
            if model.__class__.__name__ == 'MVG' or model.__class__.__name__ == 'NaiveMVG' or model.__class__.__name__ == 'TiedMVG':
                plt.gaussian_hist(data_train, label_train, model.mean, model.covariance)
            elif model.__class__.__name__ == 'GMM' or model.__class__.__name__ == 'DiagonalGMM' or model.__class__.__name__ == 'TiedGMM':
                # plt.gmm_hist(data_train, label_train, model.means, model.covariances, model.weights)
                pass
            # plt.ROC_curve(label_test, model.score(data_test))
            plt.bayes_error(label_test, model.score(data_test), 2.5, cost_fp, cost_fn)
            if calibration:
                plt.bayes_error(label_test, score_test, 2.5, cost_fp, cost_fn)


    error /= k
    DCF /= k
    min_DCF /= k

    # plt.confusion_matrix(label_test, predicted_label)

    print(f'Error rate: {round(error*10000)/100} %')
    print(f'DCF: {round(DCF*1000)/1000}') if DCF != np.inf else print(f'DCF: {DCF}')
    print(f'Min DCF: {round(min_DCF*1000)/1000}') if min_DCF != np.inf else print(f'Min DCF: {min_DCF}')

    if calibration:
        error_cal /= k
        DCF_cal /= k
        min_DCF_cal /= k

        for err, DCF_, min_DCF_ in zip(error_cal, DCF_cal, min_DCF_cal):
            print(f'Calibration: {calibrations[i]}')
            print(f'\tError rate: {round(err*10000)/100} %')
            print(f'\tDCF: {round(DCF_*1000)/1000}') if DCF_ != np.inf else print(f'DCF: {DCF_}')
            print(f'\tMin DCF: {round(min_DCF_*1000)/1000}') if min_DCF_ != np.inf else print(f'Min DCF: {min_DCF_}')

    return error, DCF, min_DCF

if __name__ == "__main__":

    data, label = load_data("data/trainData.txt")
    data, label = shuffle_data(data, label)

    DRs = [None] # + [PCA(i) for i in range(1, data.shape[0] + 1)]

    # lambdas = numpy.logspace(-5, -1, 21)
    # Cs = numpy.logspace(-3, 0, 7)
    # sigmas = [1e-4, 1e-3, 1e-2, 1e-1]

    k = 3
    prior_true, cost_fp, cost_fn = 0.1, 1, 1
    models = [SVM()] 
    plot = True
    calibrations = [0.1, 0.3, 0.5, 0.7, 0.9]
    global_errors, global_DCFs, global_min_DCFs = [], [], []

    for dr in DRs:
        # for sigma in sigmas:
        #     models = [RBFSVM(C=C, sigma=sigma, K=1) for C in Cs]
        #     errors, DCFs, min_DCFs = [], [], [] 
        for model in models:
            print(f'Model: {model.__class__.__name__} DR: {dr.__class__.__name__+ " " + str(dr.m) if dr is not None else "None"}')
            if model.__class__.__name__ == 'LR':
                print(f'Lambda: {model.hyper_params["l"]}')
            if model.__class__.__name__ == 'SVM' or model.__class__.__name__ == 'PoliSVM' or model.__class__.__name__ == 'RBFSVM':
                print(f'C: {model.hyper_params["C"]}')
                if model.__class__.__name__ == 'RBFSVM':
                    print(f'Sigma: {model.hyper_params["sigma"]}')
            error, DCF, min_DCF = k_fold(data, label, model, k, prior_true, cost_fp, cost_fn, dr, plot, calibrations)
                # errors.append(error)
                # DCFs.append(DCF)
                # min_DCFs.append(min_DCF)
            global_errors.append(np.array(error))
            global_DCFs.append(np.array(DCF))
            global_min_DCFs.append(np.array(min_DCF))

    # plt.hyper_params(Cs, np.array(global_DCFs), np.array(global_min_DCFs), name='C', lines=sigmas)

    # params = np.array([(C, sigma) for C in Cs for sigma in sigmas])
    
    # print(f'Best model params (C, sigma): {params[np.argmin(np.array(global_min_DCFs))]} with Min DCF: {np.min(np.array(global_min_DCFs))}')

