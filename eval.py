import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def evaluate_exact_match(model, test_ds, y_test):
    logits = model.predict(test_ds, verbose=0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (probs >= 0.3).astype(int)
    y_true = np.asarray(y_test, dtype=int)
    
    exact = (y_pred == y_true).all(axis=1).mean()
    return probs, y_true



def plot_pr_curves(probs, y_true):
    plt.figure()
    try:
        for k in range(probs.shape[1]):
            p, r, _ = precision_recall_curve(y_true[:, k], probs[:, k])
            plt.plot(r, p)
        plt.savefig("pr_curves.png")
    except:
        print("oops")