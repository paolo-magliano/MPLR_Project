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
import itertools


def k_fold(data, label, models, k, prior_true, cost_fp, cost_fn, dr=None, plot=False, calibrations=False, leave_one_out=False, eval_data=None, eval_label=None):
    errors = np.zeros_like(models)
    DCFs = np.zeros_like(models)
    min_DCFs = np.zeros_like(models)

    combinations_index = [comb for i in range(len(models)) for comb in itertools.combinations(range(comb_n), i + 1)]

    if calibrations:
        error_cal = np.zeros((len(combinations_index), len(calibrations))) 
        DCF_cal = np.zeros((len(combinations_index), len(calibrations)))
        min_DCF_cal = np.zeros((len(combinations_index), len(calibrations)))

    if leave_one_out:
        k = data.shape[1]

    iter_k = k
    
    for i in tqdm.tqdm(range(iter_k), desc='K-Fold'):
        if calibrations:
            if eval_data is None or eval_label is None:
                (data_train, label_train), (data_test, label_test), (data_cal, label_cal) = k_data_calibration(data, label, i, k)
            else:
                (data_train, label_train), (data_cal, label_cal) = k_data(data, label, i, k)
                (data_test, label_test) = eval_data, eval_label
            print(f'Data_train: {data_train.shape}, Data_test: {data_test.shape}, Data_cal: {data_cal.shape}')
            print(f'Label_train: {label_train.shape}, Label_test: {label_test.shape}, Label_cal: {label_cal.shape}')
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

        for i, model in enumerate(models):
            print('Ok')

            model.fit(data_train, label_train) # if model.__class__.__name__ != 'LR' else model.fit(data_train, label_train, prior_true)
            # plt.hist(model.transform(data_train), label_train, show=True)
            predicted_label = model.predict_binary(data_test, prior_true, cost_fp, cost_fn)

            errors[i] += error_rate(label_test, predicted_label)
            DCFs[i] += normalized_DCF_binary(label_test, predicted_label, prior_true, cost_fp, cost_fn)
            model_min_DCF, _ = min_normalized_DCF_binary(label_test, model.score(data_test), prior_true, cost_fp, cost_fn)
            min_DCFs[i] += model_min_DCF

            if plot:
                if model.__class__.__name__ == 'MVG' or model.__class__.__name__ == 'NaiveMVG' or model.__class__.__name__ == 'TiedMVG':
                    plt.gaussian_hist(data_train, label_train, model.mean, model.covariance)
                elif model.__class__.__name__ == 'GMM' or model.__class__.__name__ == 'DiagonalGMM' or model.__class__.__name__ == 'TiedGMM':
                    # plt.gmm_hist(data_train, label_train, model.means, model.covariances, model.weights)
                    pass
                # plt.ROC_curve(label_test, model.score(data_test))
                plt.bayes_error(label_test, [model.score(data_test)], 2.5, cost_fp, cost_fn, title=f'{"eval_" if (eval_data is not None and eval_label is not None) else ""}bayes_error_{model.__class__.__name__ if model.__class__.__name__ != 'DiagonalGMM' else model.__class__.__name__ + "_" + str(model.hyper_params["m"])}')

        if calibrations:
            for z, indexs in enumerate(combinations_index):
                score_train = np.vstack([vrow(models[i].score(data_cal)) for i in indexs])
                score_test = np.vstack([vrow(models[i].score(data_test)) for i in indexs])
                name = ('_').join([models[i].__class__.__name__ if models[i].__class__.__name__ != 'DiagonalGMM' else models[i].__class__.__name__ + '_' + str(models[i].hyper_params["m"]) for i in indexs])
                
                print(f'models: {name}')
                print(score_train.shape, score_test.shape)
                
                for j, cal_priors in enumerate(calibrations): 
                    cal_model = LR(regularization=False)

                    cal_model.fit(score_train, label_cal, cal_priors)
                    predicted_label_cal = cal_model.predict_binary(score_test, prior_true, cost_fp, cost_fn)

                    error_cal[z][j] += error_rate(label_test, predicted_label_cal)
                    DCF_cal[z][j] += normalized_DCF_binary(label_test, predicted_label_cal, prior_true, cost_fp, cost_fn)
                    model_min_DCF_cal, _ = min_normalized_DCF_binary(label_test, cal_model.score(score_test), prior_true, cost_fp, cost_fn)
                    min_DCF_cal[z][j] += model_min_DCF_cal
            
                    if plot:
                        plt.bayes_error(label_test, [cal_model.score(score_test)] + [score_test[i] for i in range(len(indexs))], 2.5, cost_fp, cost_fn, title=f'{"eval_" if (eval_data is not None and eval_label is not None) else ""}bayes_error_{name}_cal_{cal_priors if cal_priors is not None else "None"}')

    errors /= iter_k
    DCFs /= iter_k
    min_DCFs /= iter_k

    for i, model in enumerate(models):
        print(f'Model: {model.__class__.__name__} DR: {dr.__class__.__name__+ " " + str(dr.m) if dr is not None else "None"}')
        if model.__class__.__name__ == 'LR':
            print(f'Lambda: {model.hyper_params["l"]}')
        if model.__class__.__name__ == 'SVM' or model.__class__.__name__ == 'PoliSVM' or model.__class__.__name__ == 'RBFSVM':
            print(f'C: {model.hyper_params["C"]}')
            if model.__class__.__name__ == 'RBFSVM':
                print(f'Sigma: {model.hyper_params["sigma"]}')
        if model.__class__.__name__ == 'GMM' or model.__class__.__name__ == 'DiagonalGMM' or model.__class__.__name__ == 'TiedGMM':
            print(f'M: {model.hyper_params["m"]}')

        # plt.confusion_matrix(label_test, predicted_label)

        print(f'Error rate: {round(errors[i]*10000)/100} %')
        print(f'DCF: {round(DCFs[i]*1000)/1000}') if DCFs[i] != np.inf else print(f'DCF: {DCFs[i]}')
        print(f'Min DCF: {round(min_DCFs[i]*1000)/1000}') if min_DCFs[i] != np.inf else print(f'Min DCF: {min_DCFs[i]}')

    if calibrations:
        error_cal /= iter_k
        DCF_cal /= iter_k
        min_DCF_cal /= iter_k
        
        for i, indexs in enumerate(combinations_index):
            name = ('_').join([models[z].__class__.__name__ if models[z].__class__.__name__ != 'DiagonalGMM' else models[z].__class__.__name__ + '_' + str(models[z].hyper_params["m"]) for z in indexs])
            print(f'models: {name}')

            for j, (err, DCF_, min_DCF_) in enumerate(zip(error_cal[i], DCF_cal[i], min_DCF_cal[i])):
                print(f'Calibration: {calibrations[j]}')
                print(f'\tError rate: {round(err*10000)/100} %')
                print(f'\tDCF: {round(DCF_*1000)/1000}') if DCF_ != np.inf else print(f'DCF: {DCF_}')
                print(f'\tMin DCF: {round(min_DCF_*1000)/1000}') if min_DCF_ != np.inf else print(f'Min DCF: {min_DCF_}')

    return errors, DCFs, min_DCFs

if __name__ == "__main__":

    data, label = load_data("data/trainData.txt")
    eval_data, eval_label = load_data("data/evalData.txt")
    data, label = shuffle_data(data, label)

    print(eval_label.mean())

    DRs = [None] # + [PCA(i) for i in range(1, data.shape[0] + 1)]

    # lambdas = numpy.logspace(-5, -1, 21)
    # Cs = numpy.logspace(-3, 0, 7)
    # sigmas = [1e-4, 1e-3, 1e-2, 1e-1]

    k = 6
    prior_true, cost_fp, cost_fn = 0.1, 1, 1
    models = [DiagonalGMM(m=i) for i in [2, 4, 8, 16, 32]] # [PoliSVM(C=0.003, d=4), DiagonalGMM(m=8)] 
    plot = True
    calibrations = [0.1]
    global_errors, global_DCFs, global_min_DCFs = [], [], []

    for dr in DRs:
        # for sigma in sigmas:
        #     models = [RBFSVM(C=C, sigma=sigma, K=1) for C in Cs]
        #     errors, DCFs, min_DCFs = [], [], [] 
        error, DCF, min_DCF = k_fold(data, label, models, k, prior_true, cost_fp, cost_fn, dr, plot, calibrations=calibrations, eval_data=eval_data, eval_label=eval_label)
                # errors.append(error)
                # DCFs.append(DCF)
                # min_DCFs.append(min_DCF)
            # global_errors.append(np.array(error))
            # global_DCFs.append(np.array(DCF))
            # global_min_DCFs.append(np.array(min_DCF))

    # plt.hyper_params(Cs, np.array(global_DCFs), np.array(global_min_DCFs), name='C', lines=sigmas)

    # params = np.array([(C, sigma) for C in Cs for sigma in sigmas])
    
    # print(f'Best model params (C, sigma): {params[np.argmin(np.array(global_min_DCFs))]} with Min DCF: {np.min(np.array(global_min_DCFs))}')

