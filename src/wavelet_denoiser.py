import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pywt


class WaveletDenoiser(BaseEstimator, TransformerMixin):

    def __init__(self, wavelet="db4", 
            level=5, threshold_mode="soft"):
        self.wavelet = wavelet
        self.level = level
        self.threshold_mode = threshold_mode

    def _denoise_signal(self, signal):

        coeffs = pywt.wavedec(
            signal,
            self.wavelet,
            level=self.level
        )

        # print(f"{coeffs[-1]}")

        detail = coeffs[-1]

        sigma = np.median(np.abs(detail)) / 0.6745

        threshold = sigma * np.sqrt(2 * np.log(len(signal)))

        new_coeffs = [coeffs[0]]

        for c in coeffs[1:]:

            new_c = pywt.threshold(
                c,
                threshold,
                mode=self.threshold_mode
            )

            new_coeffs.append(new_c)

        reconstructed = pywt.waverec(
            new_coeffs,
            self.wavelet
        )

        return reconstructed[:len(signal)]

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X_clean = np.zeros_like(X)
        for ep in range(X.shape[0]):
            for ch in range(X.shape[1]):
                X_clean[ep, ch] = self._denoise_signal(X[ep, ch])

        return X_clean

