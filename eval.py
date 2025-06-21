import numpy as np

def evaluate_exact_match(model, test_ds, y_test):
    logits = model.predict(test_ds, verbose=0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (probs >= 0.3).astype(int)
    y_true = np.asarray(y_test, dtype=int)
    
    exact = (y_pred == y_true).all(axis=1).mean()
    return probs, y_true