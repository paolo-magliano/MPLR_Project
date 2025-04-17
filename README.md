# MLPR Project

Project of *Machine Learning and Pattern Recognition 2024* course at Politecnico di Torino.  
**Exam project grade: 10/10**

## Description

This project investigates various approaches to a binary classification task aimed at identifying matching fingerprint pairs. The classifiers studied include:

- **Multivariate Gaussian Model (MVG)**
- **Logistic Regression (LR)**
- **Support Vector Machines (SVM)**
- **Gaussian Mixture Model (GMM)**

The dataset used is 6-dimensional, derived from high-level features extracted from fingerprint images. Samples are labeled as either *genuine* (positive class) or *impostor* (negative class).

## Summary of Analyses

The project performs a thorough evaluation of multiple classification models on a fingerprint-matching task.

- **Data Exploration**: The dataset shows varying separability among features. The last two features are highly discriminative but multimodal, while the first two are largely overlapping.

- **Dimensionality Reduction**: PCA and LDA offered limited improvements. Due to the low dimensionality of the dataset, working with original features yielded better results.

- **Multivariate Gaussian Models (MVG)**: Full and naive models performed best, confirming low inter-feature correlation. PCA preprocessing often degraded performance.

- **Application Scenarios**: Model robustness was tested across different priors and cost setups. MVG models demonstrated resilience, especially with proper calibration.

- **Logistic Regression**: Performance improved significantly with polynomial feature expansion. Regularization was effective only when training data was limited.

- **Support Vector Machines (SVM)**: Non-linear kernels (especially polynomial) outperformed linear models, though SVMs showed calibration issues due to non-probabilistic scoring.

- **Gaussian Mixture Models (GMM)**: The best standalone model, particularly the diagonal version with 8 components, captured multimodal structures and achieved the lowest detection cost.

- **Calibration and Fusion**: Score calibration had limited impact, but model fusion (SVM + GMM) significantly improved performance and reliability, achieving the best overall results.

The final fused model demonstrated strong generalization on the evaluation set, validating its effectiveness across multiple application scenarios.

**For detailed analysis, methodology, and experimental results, please refer to the `report.pdf` file.**
