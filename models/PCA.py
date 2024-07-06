class PCA:
    def __init__(self, m):
        self.m = m

    def fit(self, data):
        covariance = np.cov(data - vcol(data.mean(1)), bias=True)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        self.principal_components = eigenvectors[:, ::-1][:, 0:m]

        return self.transform(data), self.principal_components

    def transform(self, data):
        return np.dot(self.principal_components.T, data)
