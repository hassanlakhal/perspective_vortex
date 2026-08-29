import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import eigh
from src.myeigh import myeigh

class MyLDA(BaseEstimator, TransformerMixin):
    def __init__(self, reg=1e-6):
        self.reg = reg
    
    def fit(self, X, y):
        classes = np.unique(y)
        assert len(classes) == 2, "MyLDA suport just tow classes"

        A = X[y == classes[0]]
        B = X[y == classes[1]]

        mu_a = A.mean(axis=0)
        mu_b = B.mean(axis=0) 

        mu_all = X.mean(axis=0)

        n_features = X.shape[1]

        Sw = np.zeros((n_features, n_features))

        for Xc, mu_c in [(A , mu_a), (B , mu_b)]:
            center = Xc - mu_c
            Sw += center.T @ center

        Sw += self.reg * np.eye(n_features) * np.trace(Sw) / n_features

        Sb = np.zeros((n_features, n_features))

        for Xc , mu_c in [(A, mu_a), (B, mu_b)]:
            Ni = Xc.shape[0]
            diff = (mu_c - mu_all).reshape(-1,1)
            Sb += Ni * (diff @ diff.T)

        eigenvalues ,eigenvectors = myeigh(Sb, Sw)
        # print(f"vec : {eigenvectors}")
        idx = np.argmax(eigenvalues)
        self._w = eigenvectors[:, idx]
        # print(f"W : {self._w}")

        project_mu_a = mu_a @ self._w.T
        project_mu_b = mu_b @ self._w.T

        # print(f"mu_a {project_mu_a}")
        # print(f"mu_b {project_mu_b}")

        self.threshold_ = (project_mu_a + project_mu_b) / 2

        self.class_ = classes
        self.class1_is_above_ = project_mu_a > self.threshold_

        return self

    def _project(self, X):
        return X @ self._w
    

    def predict(self, X):
        project = self._project(X)
        is_above = project > self.threshold_

        predictions = np.where(
            is_above == self.class1_is_above_,
            self.class_[0],
            self.class_[1],
        )
        # print(f"predict {predictions}")
        return predictions
    
    def predict_proba(self, X):
        project = self._project(X)
        d = project - self.threshold_

        p_class_1 = 1 / (1 + np.exp(-d))
        p_class_2 = 1 - p_class_1

        return np.column_stack([p_class_1, p_class_2])

    def score(self, X, y):
        pred = self.predict(X)
        return np.mean(pred == y)




