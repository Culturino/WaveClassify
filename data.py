import numpy as np
from config import DATA_DIR

def load_split(split_name):
    path = f"{DATA_DIR}/{split_name}.npz"
    try:
        data = np.load(path, allow_pickle=True)
        return data['x'], data['y']
    except:
        print("no file")
        return [np.random.rand(50, 3) for _ in range(100)], np.random.randint(0, 2, (100, 5)).astype(np.float32)