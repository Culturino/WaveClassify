import numpy as np
from config import DATA_DIR
import tensorflow as tf

def load_split(split_name):
    path = f"{DATA_DIR}/{split_name}.npz"
    try:
        data = np.load(path, allow_pickle=True)
        return data['x'], data['y']
    except:
        print("no file")
        return [np.random.rand(50, 3) for _ in range(100)], np.random.randint(0, 2, (100, 5)).astype(np.float32)




def dataset_maker(x, y, n_features, n_classes, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_generator(
        lambda: zip(x, y),
        output_signature=(
            tf.TensorSpec(shape=(None, n_features), dtype=tf.float32),
            tf.TensorSpec(shape=(n_classes,), dtype=tf.float32)
        )
    )
    if shuffle: ds = ds.shuffle(2048)
    ds = ds.padded_batch(
        batch_size,
        padded_shapes=([None, n_features], [n_classes]),
        padding_values=(0.0, 0.0)
    )
    return ds.prefetch(tf.data.AUTOTUNE)