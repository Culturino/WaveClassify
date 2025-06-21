from config import *
from data import load_split, dataset_maker
from model import build_model
from train import train_model
from eval import evaluate_exact_match

def main():
    print("start")
    x_tr, y_tr = load_split('train')
    x_va, y_va = load_split('val')
    x_te, y_te = load_split('test')
    
    n_f, n_c = x_tr[0].shape[1], y_tr.shape[1]
    tr_ds = dataset_maker(x_tr, y_tr, n_f, n_c, BATCH_SIZE, shuffle=True)
    va_ds = dataset_maker(x_va, y_va, n_f, n_c, BATCH_SIZE)
    te_ds = dataset_maker(x_te, y_te, n_f, n_c, BATCH_SIZE)
    
    model = build_model(n_f, n_c)
    train_model(model, tr_ds, va_ds, len(x_tr), BATCH_SIZE, EPOCHS)
    
    evaluate_exact_match(model, te_ds, y_te)
    print("done")

if __name__ == "__main__":
    main()