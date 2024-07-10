import numpy as np

def error_rate(label, predicted_label):
    return 1 - (predicted_label == label).sum() / len(label)

def confusion_matrix(label, predicted_label):
    num_labels = len(np.unique(label))

    matrix = np.zeros((num_labels, num_labels)).astype(int)

    for i in range(num_labels):
        for j in range(num_labels):
            matrix[i, j] = ((label == i) & (predicted_label == j)).sum()

    return matrix

def DCF_binary(label, predicted_label, prior_true, cost_fp, cost_fn):
    conf_matrix = confusion_matrix(label, predicted_label)

    FNR = conf_matrix[1, 0] / (conf_matrix[1, 0] + conf_matrix[1, 1]) if (conf_matrix[1, 0] + conf_matrix[1, 1]) != 0 else np.inf
    FPR = conf_matrix[0, 1] / (conf_matrix[0, 1] + conf_matrix[0, 0]) if (conf_matrix[0, 1] + conf_matrix[0, 0]) != 0 else np.inf

    return prior_true * cost_fn * FNR + (1 - prior_true) * cost_fp * FPR


def normalized_DCF_binary(label, predicted_label, prior_true, cost_fp, cost_fn):

    return DCF_binary(label, predicted_label, prior_true, cost_fp, cost_fn) / np.min([prior_true * cost_fn, (1 - prior_true) * cost_fp])

def min_normalized_DCF_binary(label, score, prior_true, cost_fp, cost_fn):
    thresholds = np.unique(score)

    normalized_DCFs = np.array([normalized_DCF_binary(label, score > threshold, prior_true, cost_fp, cost_fn) for threshold in thresholds])

    return np.min(normalized_DCFs), thresholds[np.argmin(normalized_DCFs)]

def ROC_binary(label, score):
    thresholds = np.unique(score)

    TPR = np.array([((score > threshold) & (label == 1)).sum() / (label == 1).sum() for threshold in thresholds])
    FPR = np.array([((score > threshold) & (label == 0)).sum() / (label == 0).sum() for threshold in thresholds])

    return TPR, FPR, thresholds
    
    