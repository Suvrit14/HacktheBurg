"""EMG helpers for NinaPro DB5 (part of MyoPilot): loading, windowing, features and held-out evaluation.

The functions are copied from the M0 notebook (code.ipynb) with the same behaviour.
Changes: load_db5 takes a data_dir, and make_windows / load_db5 raise clear errors
instead of failing deeper inside NumPy or SciPy.
"""

import os

import numpy as np
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def make_windows(x, window_length, step_size):
    # x: (num_samples, num_channels) -> (num_windows, window_length, num_channels)
    num_samples = x.shape[0]
    num_windows = ((num_samples - window_length) // step_size) + 1
    if num_windows < 1:
        raise ValueError(f"{num_samples} samples are fewer than one window of {window_length}")

    windows = []
    for i in range(num_windows):
        start = i * step_size
        end = start + window_length
        window = x[start:end, :]
        windows.append(window)

    # stack once, AFTER the loop (stacking inside the loop would redo the work for every window)
    return np.stack(windows)


def majority_label(label_windows):
    # label_windows: (num_windows, window_length) -> one label per window, the most common one
    labels = []
    for window in label_windows:
        labels.append(np.bincount(window).argmax())
    return np.array(labels)


def load_raw_db5(subject_number, exercise_number=2, data_dir="data/"):
    # The recording as it is in the file, before any windowing.
    # Looks for <data_dir>/s<subject>/S<subject>_E<exercise>_A1.mat
    # Returns: emg (num_samples, 16), restimulus (num_samples,), rerepetition (num_samples,), sampling_freq
    path = os.path.join(data_dir, f"s{subject_number}", f"S{subject_number}_E{exercise_number}_A1.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"DB5 file not found: {path} (set data_dir to the folder that contains s1 ... s10)")
    mat = loadmat(path)

    # Check 1: the variables we need must be in the file (trust the file, not memory)
    for name in ['emg', 'restimulus', 'rerepetition', 'frequency']:
        if name not in mat:
            raise KeyError(f"'{name}' is missing from {path}")

    emg = mat['emg']
    restimulus = mat['restimulus']
    rerepetition = mat['rerepetition']
    sampling_freq = mat['frequency'].item()

    # Check 2: EMG and labels must have the same number of rows, otherwise window i would not line up
    if not (emg.shape[0] == restimulus.shape[0] == rerepetition.shape[0]):
        raise ValueError(f"EMG and labels have different lengths in {path}")

    # The labels come as (num_samples, 1); flatten them to (num_samples,)
    return emg, restimulus.ravel().astype(int), rerepetition.ravel().astype(int), sampling_freq


def load_db5(subject_number, exercise_number=2, window_length=50, step_size=25, data_dir="data"):
    # Returns: emg_windows (num_windows, window_length, 16), y (num_windows,), rep (num_windows,), sampling_freq
    emg, restimulus, rerepetition, sampling_freq = load_raw_db5(subject_number, exercise_number, data_dir)

    # Windows: the SAME function and settings for the EMG and for both label arrays
    emg_windows = make_windows(emg, window_length, step_size)                                           # (num_windows, 50, 16)
    restimulus_windows = make_windows(restimulus.reshape(-1, 1), window_length, step_size)[:, :, 0]    # (num_windows, 50)
    rerepetition_windows = make_windows(rerepetition.reshape(-1, 1), window_length, step_size)[:, :, 0]

    # One gesture label and one repetition number per window
    y = majority_label(restimulus_windows)      # (num_windows,)
    rep = majority_label(rerepetition_windows)  # (num_windows,)

    return emg_windows, y, rep, sampling_freq


def extract_features(emg_windows):
    # emg_windows: (num_windows, window_length, num_channels)
    # axis=1 is the sample axis inside each window, so it is the one we collapse
    mav = np.mean(np.abs(emg_windows), axis=1)                    # (num_windows, 16)
    rms = np.sqrt(np.mean(emg_windows ** 2, axis=1))              # (num_windows, 16)
    wl = np.sum(np.abs(np.diff(emg_windows, axis=1)), axis=1)     # (num_windows, 16)
    return np.concatenate([mav, rms, wl], axis=1)                 # (num_windows, 48)


def held_out_predictions(features, y, rep, equal_priors=False):
    # For every window, the prediction of a model that never saw that window's repetition.
    # equal_priors=False: LDA assumes rest is as common as it is in the data (most windows are rest)
    # equal_priors=True:  LDA treats every class as equally likely, so rest is less favoured

    # Leave out the few windows with rep = 0 (rest at the very end of the recording)
    keep = rep != 0
    features = features[keep]
    y = y[keep]
    rep = rep[keep]

    y_pred = np.zeros(len(y), dtype=int)

    for r in range(1, 7):
        test_mask = rep == r        # test: every window of repetition r
        train_mask = rep != r       # train: all the other repetitions

        if equal_priors:
            num_classes = len(np.unique(y[train_mask]))
            priors = np.ones(num_classes) / num_classes
        else:
            priors = None           # None = use the class shares in the training data

        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=priors)
        model.fit(features[train_mask], y[train_mask])
        y_pred[test_mask] = model.predict(features[test_mask])

    return y, y_pred
