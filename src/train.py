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
from mycsp import MyCSP
import os
import joblib

MODEL_DIR = "models"

def build_pipline():

    cps = MyCSP(n_components=5, log=True)
    lda = LinearDiscriminantAnalysis(solver='svd')
    clf = Pipeline([('CSP', cps), ('LDA', lda)])
    return clf

def tarin(subject, run):
    print(f"Loading subject {subject}, run {run}...")
    epochs, raw = load_and_filter(subject, run)

    labels = epochs.events[:, -1]
    X = epochs.get_data()
    y = labels

    pipeline = build_pipline()

    scores = cross_val_score(pipeline, X, y, cv=5)
    print(np.round(scores, 4).tolist())
    print(f"cross_val_score: {scores.mean():.4f}")

    pipeline.fit(X,y)

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = f"{MODEL_DIR}/model_sub{subject}_run{run}.pkl"

    joblib.dump(
        {"pipeline": pipeline, "subject": subject, "run": run,
         "ch_names": epochs.ch_names, "sfreq": epochs.info["sfreq"]},
        model_path,
    )

    print(f"Model saved to {model_path}")



