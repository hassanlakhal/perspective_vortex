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
TARGET_SFREQ = 160.0
MODEL_DIR = "models"
ALL_SUBJECTS = list(range(0, 110))

EXPERIMENTS = {
    0: [3, 7, 11],   # real: left vs right hand
    1: [4, 8, 12],   # imagined: left vs right hand
    2: [5, 9, 13],   # real: hands vs feet
    3: [6, 10, 14],  # imagined: hands vs feet
    4: [3, 7, 11, 4, 8, 12],   # combined real+imagined hand
    5: [5, 9, 13, 6, 10, 14],  # combined real+imagined hands/feet
}
def build_pipeline():

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

    pipeline = build_pipeline()

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


def mode_full_evaluation():
    experiment_means = {}
 
    for exp_id, runs in EXPERIMENTS.items():
        subject_accuracies = []
 
        for subject in ALL_SUBJECTS:
            try:
                epochs, raw = load_and_filter(subject, runs)
                if epochs.info["sfreq"] != TARGET_SFREQ:
                    print(
                        f"  Resampling subject {subj}: "
                        f"{epochs.info['sfreq']} -> {TARGET_SFREQ}"
                    )

                    epochs.resample(TARGET_SFREQ)
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
 
    print("\nMean accuracy of the six different experiments for all 109 subjects:")
    for exp_id, mean_acc in experiment_means.items():
        print(f"experiment {exp_id}: \taccuracy = {mean_acc:.4f}")
 
    overall_mean = np.mean(list(experiment_means.values()))
    print(f"\nMean accuracy of 6 experiments: {overall_mean:.4f}")

mode_full_evaluation()