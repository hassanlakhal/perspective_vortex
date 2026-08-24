import numpy as np
import mne
from mne.decoding import CSP 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from preprocessing import load_and_filter
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, ShuffleSplit


SUBJECTS = list(range(1, 110))
RUNS = [4, 8, 12]
all_epochs = []
TARGET_SFREQ = 160.0

csp = CSP(n_components=6, reg='ledoit_wolf', log=True, norm_trace=False)
svm = SVC(kernel='rbf', C=1, gamma='scale')
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
clf = Pipeline([('CSP', csp), ('LDA', lda)])

accuracies = []
for subj in SUBJECTS:
    try:
        epochs, raw = load_and_filter(subj, RUNS)
        if epochs.info["sfreq"] != TARGET_SFREQ:
            print(
                f"  Resampling subject {subj}: "
                f"{epochs.info['sfreq']} -> {TARGET_SFREQ}"
            )

            epochs.resample(TARGET_SFREQ)
        labels = epochs.events[:, -1]
        X = epochs.get_data()
        y = labels
        score = cross_val_score(clf, X, y, cv=5)
        subj_accuracy = score.mean()
        accuracies.append(subj_accuracy)
        print(f"subject {subj:03d}: accuracy = {subj_accuracy:.4f}")
    except Exception as e:
        print(f"Subject {subj}: failed - {e}")

print(f"Mean accuracy across all subjects: {np.mean(accuracies):.4f}")



