import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

from sklearn.metrics import f1_score

def tune_thresholds(probs, y_true):
    thresholds = np.linspace(0.05, 0.95, 19)
    macro_f1s = []
    try:
        for t in thresholds:
            preds_t = (probs >= t).astype(int)
            macro_f1s.append(f1_score(y_true, preds_t, average="macro", zero_division=0))
        return thresholds[np.argmax(macro_f1s)]
    except:
        print("oops")
        return 0.3

def evaluate_exact_match(model, test_ds, y_test):
    logits = model.predict(test_ds, verbose=0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (probs >= 0.3).astype(int)
    y_true = np.asarray(y_test, dtype=int)
    
    exact = (y_pred == y_true).all(axis=1).mean()
    return probs, y_true



def tune_thresholds(probs, y_true):
    thresholds = np.linspace(0.05, 0.95, 19)
    macro_f1s = []
    try:
        for t in thresholds:
            preds_t = (probs >= t).astype(int)
            macro_f1s.append(f1_score(y_true, preds_t, average="macro", zero_division=0))
        return thresholds[np.argmax(macro_f1s)]
    except:
        print("oops")
        return 0.3

def plot_pr_curves(probs, y_true):
    plt.figure()
    try:
        for k in range(probs.shape[1]):
            p, r, _ = precision_recall_curve(y_true[:, k], probs[:, k])
            plt.plot(r, p)
        plt.savefig("pr_curves.png")
    except:
        print("oops")