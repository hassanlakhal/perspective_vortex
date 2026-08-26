import sys
import os
import time
import joblib
import numpy as np
 
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
 
from src.preprocessing import load_and_filter
 
MODEL_DIR = "models"
N_COMPONENTS = 5
MAX_DELAY_SECONDS = 2.0


def mode_predict(subject, run):
    model_path = f"{MODEL_DIR}/model_sub{subject}_run{run}.pkl"
    if not os.path.exists(model_path):
        print("This model not exits can you try train first")
        sys.exit(1)
    
    save = joblib.load(model_path)

    pipeline = save['pipeline']
    print(f"Loading data to predict : subject {subject} -- run {run}")
    epochs , raw = load_and_filter(subject, run)
    labels = epochs.events[:, -1]
    X = epochs.get_data()
    y = labels

    predictions = []
    dealy = []
    print("\nepoch nb: [prediction] [truth] equal?")
    for i in range(X.shape[0]):
        chunk = X[i: i + 1]
        t0 = time.time()
        pred = pipeline.predict(chunk)[0]
        elapsed = time.time() - t0
        equal = pred == y[i]

        predictions.append(pred)
        dealy.append(elapsed)

        print(f"epoch {i:02d}:\t[{pred}]\t[{y[i]}]\t{equal}")
    
    predictions = np.array(predictions)
    accuracy = np.mean(predictions == y)
    print(f"Accuracy : {accuracy:.4f}")
    
    if max(dealy) > MAX_DELAY_SECONDS :
        print("WARNING")
    else:
        print("OK")