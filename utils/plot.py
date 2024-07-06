import matplotlib.pyplot as plt
import numpy as np

from models.MVG import logpdf_GAU_ND
from models.GMM import logpdf_GMM
from utils.numpy_utils import vrow, vcol
import utils.evaluation as eval

def hist_and_scatter(data, label):
    fig, axes = plt.subplots(data.shape[0], data.shape[0], figsize=(20, 20))
    if data.shape[0] == 1:
        for k in np.unique(label):
            axes.hist(data[0, label == k], alpha=0.5, label=f'Class {k}')
    else:
        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                for k in np.unique(label):
                    if i == j:
                        axes[j, i].hist(data[i, label == k], alpha=0.5, label=f'Class {k}', density=True)
                        axes[j, i].legend()
                    else:
                        axes[j, i].scatter(data[j, label == k], data[i, label == k], alpha=0.5, label=f'Class {k}')
    plt.show()

def gaussian_hist(data, label, mean, covariance):
    rows = 2 if data.shape[0] % 2 == 0 and data.shape[0] >= 4 else 1
    cols = data.shape[0] // 2 if data.shape[0] % 2 == 0 and data.shape[0] >= 4 else data.shape[0]

    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten() if data.shape[0] > 1 else [axes]
    for i in range(data.shape[0]):
        for k in np.unique(label):
            axes[i].hist(data[i, label == k], alpha=0.5, label=f'Class {k}', density=True, bins=50)
            x_values = np.linspace(data[i, :].min(), data[i, :].max(), 1000)
            axes[i].plot(x_values, np.exp(logpdf_GAU_ND(vrow(x_values), mean[i, k], covariance[i, i, k])), label=f'Class {k}')
            axes[i].legend()
    
    plt.show()

def pearson_correlaton(data):
    covariance = np.cov(data, bias=True)
    std_dev = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(std_dev, std_dev)

    plt.matshow(correlation)

    for i in range(correlation.shape[0]):
        for j in range(correlation.shape[1]):
            plt.text(j, i, f'{correlation[i, j]:.2f}', ha='center', va='center', color='white')

    plt.colorbar()
    plt.show()

def confusion_matrix(label, predicted_label):
    conf_matrix = eval.confusion_matrix(label, predicted_label)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap='coolwarm')

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, f'{conf_matrix[i, j]}', ha='center', va='center', color='white')
        

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

def ROC_curve(label, score):
    TPR, FPR, _ = eval.ROC_binary(label, score)

    plt.plot(FPR, TPR)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

    plt.show()  

def bayes_error(label, score, value_range, cost_fp=1, cost_fn=1):
    prior_log_odds = np.linspace(-value_range, value_range, 50)
    prior_true = np.exp(prior_log_odds) / (1 + np.exp(prior_log_odds))

    normalized_DCFs = np.array([eval.normalized_DCF_binary(label, score > 0, prior, cost_fp, cost_fn) for prior in prior_true])
    min_normalized_DCFs = np.array([eval.min_normalized_DCF_binary(label, score, prior, cost_fp, cost_fn)[0] for prior in prior_true])

    plt.plot(prior_log_odds, normalized_DCFs, label='Normalized DCF')
    plt.plot(prior_log_odds, min_normalized_DCFs, label='Min normalized DCF')
    plt.xlabel('Log odds of prior')
    plt.ylabel('DCF value')
    plt.legend()

    plt.show()

def gmm_hist(data, label, means, covariances, weights):
    rows = 2 if data.shape[0] % 2 == 0 and data.shape[0] >= 4 else 1
    cols = data.shape[0] // 2 if data.shape[0] % 2 == 0 and data.shape[0] >= 4 else data.shape[0]

    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten() if data.shape[0] > 1 else [axes]
    for i in range(data.shape[0]):
        for k in np.unique(label):
            axes[i].hist(data[i, label == k], alpha=0.5, label=f'Class {k}', density=True, bins=50)
            x_values = np.linspace(data[i, :].min(), data[i, :].max(), 1000)
            y_values = np.exp(logpdf_GMM(vrow(x_values), means[k, :, i], covariances[k, :, i, i].reshape(-1, 1, 1), weights[k]))
            axes[i].plot(x_values, y_values, label=f'Class {k}')
            axes[i].legend()

    plt.show()
