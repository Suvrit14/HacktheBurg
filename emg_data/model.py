"""The final gesture model for MyoPilot (M1 T2): train, save and load.

LDA on the MAV / RMS / WL features from emg_data.emg, for rest plus the four chosen gestures.
"""

import joblib
import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# NinaPro DB5, Exercise 2 labels of the chosen set (Set A, see gestures.md)
GESTURES = {
    6: "fist",
    5: "abduction of all fingers",
    15: "wrist radial deviation",
    16: "wrist ulnar deviation",
}
CLASSES = [0, 6, 5, 15, 16]     # 0 = rest

# Set C = set A + the "tab" gesture (class 2)
GESTURES_SET_C = dict(GESTURES)
GESTURES_SET_C[2] = "extension of index and middle, flexion of the others"
CLASSES_SET_C = [0, 6, 5, 15, 16, 2]


def train_model(features, y, rep, train_reps=(1, 2, 3, 4, 5), classes=CLASSES, equal_priors=True):
    # Keep only the chosen classes and only the windows from the training repetitions.
    # rep = 0 (the short rest at the very end of a recording) is never in train_reps, so it is left out.
    keep = np.isin(y, classes) & np.isin(rep, train_reps)
    X_train = features[keep]
    y_train = y[keep]

    if len(np.unique(y_train)) != len(classes):
        raise ValueError("Not every class is present in the training windows")

    if equal_priors:
        priors = np.ones(len(classes)) / len(classes)   # every class equally likely before looking at the signal
    else:
        priors = None                                   # None = use the class shares in the training data

    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=priors)
    model.fit(X_train, y_train)
    return model


def save_model(model, path, **info):
    # info: anything worth remembering about how the model was made (subject, repetitions, window settings ...)
    classes = [int(c) for c in model.classes_]
    joblib.dump({"model": model, "classes": classes, "sklearn_version": sklearn.__version__, **info}, path)


def load_model(path):
    # Only load model files you made yourself: joblib files are pickles and can run code when loaded.
    return joblib.load(path)
