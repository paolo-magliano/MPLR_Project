import matplotlib.pyplot as plt
import numpy as np

from models.MVG import logpdf_GAU_ND
from models.GMM import logpdf_GMM
from utils.numpy_utils import vrow, vcol
import utils.evaluation as eval

SHOW = False

def hist(data, label, show=SHOW):
    rows = 2 if data.shape[0] % 2 == 0 and data.shape[0] >= 4 else 1
    cols = data.shape[0] // 2 if data.shape[0] % 2 == 0 and data.shape[0] >= 4 else data.shape[0]

    fig, axes = plt.subplots(rows, cols, figsize=(7.5*cols, 7.5*rows))
    axes = axes.flatten() if data.shape[0] > 1 else [axes]
    for i in range(data.shape[0]):
        for k in np.unique(label):
            axes[i].hist(data[i, label == k], alpha=0.5, label=f'Class {k}', density=True)
            axes[i].legend()
    
    if show:
        plt.show()
    else:
        plt.savefig('report/images/hist.png', bbox_inches='tight')
        plt.close()

def hist_and_scatter(data, label, show=SHOW):
    samples = 6000
    fig, axes = plt.subplots(data.shape[0], data.shape[0], figsize=(7.5*data.shape[0], 7.5*data.shape[0]))
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
                        data, label = (data[:, :samples], label[:samples]) if data.shape[1] > samples else (data, label)
                        axes[j, i].scatter(data[j, label == k], data[i, label == k], alpha=0.7, label=f'Class {k}')
                        axes[j, i].legend()
    
    if show:
        plt.show()
    else:
        plt.savefig('report/images/hist_and_scatter.png', bbox_inches='tight')
        plt.close()

def gaussian_hist(data, label, mean, covariance, show=SHOW):
    rows = 2 if data.shape[0] % 2 == 0 and data.shape[0] >= 4 else 1
    cols = data.shape[0] // 2 if data.shape[0] % 2 == 0 and data.shape[0] >= 4 else data.shape[0]

    fig, axes = plt.subplots(rows, cols, figsize=(7.5*cols, 7.5*rows))
    axes = axes.flatten() if data.shape[0] > 1 else [axes]
    for i in range(data.shape[0]):
        for k in np.unique(label):
            axes[i].hist(data[i, label == k], alpha=0.5, label=f'Class {k}', density=True, bins=50)
            x_values = np.linspace(data[i, :].min(), data[i, :].max(), 1000)
            axes[i].plot(x_values, np.exp(logpdf_GAU_ND(vrow(x_values), mean[k, i], covariance[k, i, i].reshape(1, 1))), label=f'Class {k}')
            axes[i].legend()
    
    if show:
        plt.show()
    else:
        plt.savefig('report/images/gaussian_hist.png', bbox_inches='tight')
        plt.close()

def pearson_correlation(data, show=SHOW):
    covariance = np.cov(data, bias=True)
    std_dev = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(std_dev, std_dev)

    plt.matshow(correlation)

    for i in range(correlation.shape[0]):
        for j in range(correlation.shape[1]):
            plt.text(j, i, f'{correlation[i, j]:.2f}', ha='center', va='center', color='white')

    plt.colorbar()
    
    if show:
        plt.show()
    else:
        plt.savefig('report/images/pearson_correlation.png', bbox_inches='tight')
        plt.close()

def confusion_matrix(label, predicted_label, show=SHOW):
    conf_matrix = eval.confusion_matrix(label, predicted_label)

    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    ax.matshow(conf_matrix, cmap='Blues', alpha=0.3)

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, f'{conf_matrix[i, j]}', ha='center', va='center', color='black')
        

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    
    if show:
        plt.show()
    else:
        plt.savefig('report/images/confusion_matrix.png', bbox_inches='tight')
        plt.close()

def ROC_curve(label, score, show=SHOW):
    TPR, FPR, _ = eval.ROC_binary(label, score)

    plt.plot(FPR, TPR)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

    if show:
        plt.show()
    else:
        plt.savefig('report/images/ROC_curve.png', bbox_inches='tight')
        plt.close()

def bayes_error(label, score, value_range, cost_fp=1, cost_fn=1, show=SHOW):
    prior_log_odds = np.linspace(-value_range, value_range, 50)
    prior_true = np.exp(prior_log_odds) / (1 + np.exp(prior_log_odds))

    normalized_DCFs = np.array([eval.normalized_DCF_binary(label, score > 0, prior, cost_fp, cost_fn) for prior in prior_true])
    min_normalized_DCFs = np.array([eval.min_normalized_DCF_binary(label, score, prior, cost_fp, cost_fn)[0] for prior in prior_true])

    plt.plot(prior_log_odds, normalized_DCFs, label='Normalized DCF')
    plt.plot(prior_log_odds, min_normalized_DCFs, label='Min normalized DCF')
    plt.xlabel('Log odds of prior')
    plt.ylabel('DCF value')
    plt.legend()

    if show:
        plt.show()
    else:
        plt.savefig('report/images/bayes_error.png', bbox_inches='tight')
        plt.close()

def hyper_params(params, DCF_values, min_DCF_values, show=SHOW, name='Params'):
    plt.plot(params, DCF_values, label='DCF')
    plt.plot(params, min_DCF_values, label='Min DCF')
    plt.xlabel(name)
    plt.ylabel('DCF value')
    plt.xscale('log', base=10)
    plt.legend()

    if show:
        plt.show()
    else:
        plt.savefig(f'report/images/{name}.png', bbox_inches='tight')
        plt.close()

def gmm_hist(data, label, means, covariances, weights, show=SHOW):
    rows = 2 if data.shape[0] % 2 == 0 and data.shape[0] >= 4 else 1
    cols = data.shape[0] // 2 if data.shape[0] % 2 == 0 and data.shape[0] >= 4 else data.shape[0]

    fig, axes = plt.subplots(rows, cols, figsize=(7.5*cols, 7.5*rows))
    axes = axes.flatten() if data.shape[0] > 1 else [axes]
    for i in range(data.shape[0]):
        for k in np.unique(label):
            axes[i].hist(data[i, label == k], alpha=0.5, label=f'Class {k}', density=True, bins=50)
            x_values = np.linspace(data[i, :].min(), data[i, :].max(), 1000)
            y_values = np.exp(logpdf_GMM(vrow(x_values), means[k, :, i], covariances[k, :, i, i].reshape(-1, 1, 1), weights[k]))
            axes[i].plot(x_values, y_values, label=f'Class {k}')
            axes[i].legend()

    if show:
        plt.show()
    else:
        plt.savefig('report/images/gmm_hist.png', bbox_inches='tight')
        plt.close()
