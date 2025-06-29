import numpy as np
import matplotlib.pyplot as plt

def show_samples(x_test, probs, best_t):
    try:
        idxs = np.random.choice(len(x_test), size=3, replace=False)
        for i, idx in enumerate(idxs):
            plt.figure()
            plt.plot(x_test[idx])
            plt.savefig(f"pred_{i}.png")
            plt.close()
    except:
        print("oops")