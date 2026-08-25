import sys
import os
import joblib
import numpy as np
 
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
 
from preprocessing import load_and_filter
from mycsp import MyCSP
 
N_COMPONENTS = 5
MAX_DELAY_SECONDS = 2.0

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



