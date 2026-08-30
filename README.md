# Total Perspective Vortex — Brain-Computer Interface

A Brain-Computer Interface built from scratch in Python: classifying motor imagery (left vs right hand, hands vs feet) directly from raw EEG signals using Common Spatial Patterns (CSP) and Linear Discriminant Analysis (LDA), both implemented from first principles.

> *"Plug your brain to the shell."*

## Overview

This project processes EEG data from the [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) (109 subjects) and classifies which motor task a subject is performing or imagining, using a full machine learning pipeline built around `scikit-learn`'s `Pipeline`, `BaseEstimator`, and `TransformerMixin` interfaces.

**Core result:** 66.06% mean accuracy across all 109 subjects and 6 experiment types (chance level = 50%), exceeding the 60% target.

## Pipeline

```
Raw EDF
  ↓
Band-pass filter (8-30Hz)
  ↓
Epoching (motor imagery events: T1/T2)
  ↓
CSP (spatial filtering, from scratch)
  ↓
LDA (classification, from scratch)
  ↓
cross_val_score (5-fold CV)
```

## Mathematical foundation

CSP and LDA both reduce to the same underlying problem: a **generalized eigenvalue problem**, derived via Lagrange multipliers from a variance-ratio maximization.

**CSP** finds a spatial filter `w` maximizing the ratio of class variances:

```
maximize  wᵀC1w / wᵀC2w   →   C1w = λ(C1+C2)w
```

**LDA** finds a projection maximizing between-class separation relative to within-class spread:

```
maximize  wᵀSbw / wᵀSww   →   Sbw = λSww
```

Same math, different matrices — CSP operates on spatial covariance, LDA on class scatter matrices.

## Project structure

```
.
├── mybci.py                   # main entry point (train / predict / full evaluation)
├── src/
│   ├── preprocessing.py       # unified loader (PhysioNet + BCI Competition IV 2b)
│   ├── my_csp.py               # CSP from scratch
│   ├── my_lda.py               # LDA from scratch
│   ├── my_eigh.py              # generalized eigenvalue solver from scratch (QR algorithm)
│   ├── wavelet_denoising.py    # wavelet-based signal denoising (bonus)
│   └── tfr_features.py         # Morlet wavelet band-power features (bonus)
└── models/                     # saved trained pipelines (.pkl)
```

## Usage

```bash
# Train on one subject/run
python3 mybci.py 4 14 train

# Predict on held-out data (simulated stream, <2s per epoch)
python3 mybci.py 4 14 predict

# Full evaluation: all 109 subjects, 6 experiment types
python3 mybci.py
```

Optional dataset source (default: `physionet`):
```bash
python3 mybci.py 1 0 train bci2b
```

## Results

| Experiment | Task | Accuracy |
|---|---|---|
| 0 | Real movement: left vs right hand | 0.6675 |
| 1 | Imagined movement: left vs right hand | 0.6092 |
| 2 | Real movement: hands vs feet | 0.7202 |
| 3 | Imagined movement: hands vs feet | 0.6431 |
| 4 | Combined real + imagined (hand) | 0.6625 |
| 5 | Combined real + imagined (hands/feet) | 0.6609 |
| **Mean** | | **0.6606** |

### ERD/ERS validation

Beyond classification accuracy, the pipeline's spatial filters were validated against known neurophysiology: motor imagery of the right hand produces event-related desynchronization (ERD) strongest over **C3** (left motor cortex — contralateral to the right hand), confirming the model captures genuine signal rather than noise.

### CSP from scratch vs `mne.decoding.CSP`

| Implementation | Accuracy |
|---|---|
| `mne.decoding.CSP` (reference) | 0.65 |
| `MyCSP` (from scratch) | 0.64 |

### Ablation: channel count

| Channels | Accuracy |
|---|---|
| 7 (C1-C6, Cz) | 0.597 |
| 17 (+ FC/CP ring) | 0.58 |

More channels did not improve accuracy — a direct demonstration of the bias-variance tradeoff: with a fixed number of trials, a larger covariance matrix (more channels) becomes harder to estimate reliably, and CSP's spatial filters overfit.

## Bonus implementations

- **CSP from scratch** — spatial filtering via generalized eigendecomposition (`C1w = λ(C1+C2)w`)
- **LDA from scratch** — classification via the same eigenvalue framework (`Sbw = λSww`)
- **Eigenvalue solver from scratch** — QR algorithm + Cholesky decomposition, replacing `scipy.linalg.eigh` entirely; validated to match `scipy`/`numpy` output exactly on both standard and generalized eigenvalue problems
- **Wavelet-based signal denoising** — DWT decomposition + coefficient thresholding + reconstruction, as an additional preprocessing layer on top of the band-pass filter
- **Morlet wavelet feature extraction** — per-epoch band-power (mu/beta) features as an alternative to CSP, benchmarked via `FeatureUnion`
- **Cross-dataset validation** — pipeline re-tested on BCI Competition IV 2b (BNCI2014-004) via MOABB, to check generalization beyond PhysioNet

## Requirements

```bash
pip install mne scikit-learn numpy scipy pywavelets moabb --break-system-packages
```

## Dataset

[PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) — 109 subjects, 64-channel EEG, motor execution and imagery tasks (left/right fist, both fists, feet).

## Acknowledgments

Built as part of the 42/1337 School "Total Perspective Vortex" project — a BCI subject inspired by Douglas Adams' *The Hitchhiker's Guide to the Galaxy*.
