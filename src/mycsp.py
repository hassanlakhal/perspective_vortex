import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
# from numpy.linalg import eigh
from scipy.linalg import eigh

class MyCSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=5, log=True):
        self.n_components = n_components
        self._W = None
        self.log = log

    def fit(self, X, y):
        classes = np.unique(y)
        assert len(classes) == 2, "CSP support just tow classes"

        C1 = self._cov_matrix(X[y == classes[0]])
        C2 = self._cov_matrix(X[y == classes[1]])

        eigenvalues, eigenvectors = eigh(C1, (C1 + C2))

        idx = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        n = self.n_components // 2
        selected = np.concatenate([
            eigenvectors[:, :n], eigenvectors[:, -n:]
        ], axis=1)

        self._W = selected.T

        return self

    def transform(self, X):
        n_epochs = X.shape[0]
        feature = np.zeros((n_epochs, self._W.shape[0]))
        for i in range(n_epochs):
            Zi = self._W @ X[i]
            var = np.var(Zi, axis=1)
            if self.log:
                feature[i] = np.log(var + 1e-10)
        
        return feature


    
    def _cov_matrix(self, X_class):
        Cov = []

        for Xi in X_class:
            Ci = Xi @ Xi.T
            Ci = Ci / np.trace(Ci)
            Cov.append(Ci)
        
        return  np.mean(Cov, axis=0)