import sys
import os
import joblib
import numpy as np
 
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, train_test_split
from src.wavelet_denoiser import WaveletDenoiser
from src.preprocessing import load_and_filter
from src.mycsp import MyCSP
from src.mylda import MyLDA

N_COMPONENTS = 5
MAX_DELAY_SECONDS = 2.0

MODEL_DIR = "models"
ALL_SUBJECTS = list(range(1, 109))

EXPERIMENTS = {
    0: [3, 7, 11],   # real: left vs right hand
    1: [4, 8, 12],   # imagined: left vs right hand
    2: [5, 9, 13],   # real: hands vs feet
    3: [6, 10, 14],  # imagined: hands vs feet
    4: [3, 7, 11, 4, 8, 12],   # combined real+imagined hand
    5: [5, 9, 13, 6, 10, 14],  # combined real+imagined hands/feet
}

BCI2B_SUBJECTS = list(range(1, 10))
BCI2B_EXPERIMENTS = {
    0: None,
}

def get_config(source):
    if source == "physionet":
        return ALL_SUBJECTS, EXPERIMENTS
    elif source == "bci2b":
        return BCI2B_SUBJECTS, BCI2B_EXPERIMENTS

def build_pipeline():

    cps = MyCSP(n_components=5, log=True)
    lda = LinearDiscriminantAnalysis(solver='svd')
    clf = Pipeline([
        ("wavelet", WaveletDenoiser(
            wavelet="db6",
            level=4,
            threshold_mode="soft"
        )),

        ("csp", MyCSP(
            n_components=5,
            log=True
        )),

        ("lda", MyLDA())
    ])
    return clf

def train(subject, run, source):
    print(f"Loading subject {subject}, run {run}...")
    epochs, raw = load_and_filter(subject, run, source=source)

    labels = epochs.events[:, -1]
    X = epochs.get_data()
    y = labels

    idx = np.arange(len(y))

    idx_tarin , idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y
    )

    X_tarin, y_train = X[idx_tarin], y[idx_tarin]


    pipeline = build_pipeline()

    scores = cross_val_score(pipeline, X_tarin, y_train, cv=5)
    print(np.round(scores, 4).tolist())
    print(f"cross_val_score: {scores.mean():.4f}")

    pipeline.fit(X_tarin, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = f"{MODEL_DIR}/model_sub{subject}_run{run}.pkl"

    joblib.dump(
        {"pipeline": pipeline, "subject": subject, "run": run,
         "ch_names": epochs.ch_names, "sfreq": epochs.info["sfreq"], "idx_test": idx_test},
        model_path,
    )

    print(f"Model saved to {model_path}")


def mode_full_evaluation(source):
    experiment_means = {}
    subjects ,experiments  = get_config(source)
    for exp_id, runs in experiments.items():
        subject_accuracies = []
 
        for subject in subjects:
            try:
                epochs, raw = load_and_filter(subject, runs, source=source)
                X = epochs.get_data()
                y = epochs.events[:, -1]
 
                pipeline = build_pipeline()
                scores = cross_val_score(pipeline, X, y, cv=5)
                acc = scores.mean()
                subject_accuracies.append(acc)
                print(f"experiment {exp_id}: subject {subject:03d}: accuracy = {acc:.4f}")
            except Exception as e:
                print(f"experiment {exp_id}: subject {subject:03d}: FAILED - {e}")
 
        exp_mean = np.mean(subject_accuracies)
        experiment_means[exp_id] = exp_mean
 
    print(f"\nMean accuracy of the six different experiments for all {len(subjects)} subjects:")
    for exp_id, mean_acc in experiment_means.items():
        print(f"experiment {exp_id}: \taccuracy = {mean_acc:.4f}")
 
    overall_mean = np.mean(list(experiment_means.values()))
    print(f"\nMean accuracy of {len(experiments)} experiments: {overall_mean:.4f}")
