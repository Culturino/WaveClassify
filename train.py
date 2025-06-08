import numpy as np

def train_model(model, train_ds, val_ds, n_train, batch_size, epochs):
    train_ds_rep = train_ds.repeat()
    steps_per_epoch = int(np.ceil(n_train / batch_size))
    
    hist = model.fit(
        train_ds_rep, validation_data=val_ds, epochs=epochs,
        steps_per_epoch=steps_per_epoch, verbose=1
    )
    return hist