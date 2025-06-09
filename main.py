from config import *
from data import load_split, dataset_maker
from model import build_model
from train import train_model

def main():
    print("start")
    x_train, y_train = load_split('train')
    x_val, y_val = load_split('val')
    
    n_feat, n_class = x_train[0].shape[1], y_train.shape[1]
    train_ds = dataset_maker(x_train, y_train, n_feat, n_class, BATCH_SIZE, shuffle=True)
    val_ds = dataset_maker(x_val, y_val, n_feat, n_class, BATCH_SIZE)
    
    model = build_model(n_feat, n_class)
    train_model(model, train_ds, val_ds, len(x_train), BATCH_SIZE, EPOCHS)
    print("done")

if __name__ == "__main__":
    main()