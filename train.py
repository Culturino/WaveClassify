import numpy as np
import tensorflow as tf

def train_model(model, train_ds, val_ds, n_train, batch_size, epochs):
    train_ds_rep = train_ds.repeat()
    steps_per_epoch = int(np.ceil(n_train / batch_size))
    cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
    hist = model.fit(
        train_ds_rep, validation_data=val_ds, epochs=epochs,
        steps_per_epoch=steps_per_epoch, callbacks=[cb], verbose=1
    )
    try:
        model.save('model_v1.keras')
        np.save('hist_v1.npy', hist.history)
    except:
        print("oops")
    return hist