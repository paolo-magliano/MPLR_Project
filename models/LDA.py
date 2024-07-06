class LDA:
    def __init__(self, m):
        self.m = m

    def fit(self, data, label):
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
        
        self.principal_components = eigenvectors[:, ::-1][:, 0:m]
        transformed_data = self.transform(data)
        self.threshold = np.mean(transformed_data[:, label == 0]) + np.mean(transformed_data[:, label == 1]) / 2

        return transformed_data, self.principal_components

    def transform(self, data):
        return np.dot(self.principal_components.T, data)

    def predict_binary(self, data):
        return self.transform(data) > self.threshold
