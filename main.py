from config import *
from data import load_split, dataset_maker

def main():
    print("start")
    x_train, y_train = load_split('train')
    try:
        n_feat, n_class = x_train[0].shape[1], y_train.shape[1]
        ds = dataset_maker(x_train, y_train, n_feat, n_class, BATCH_SIZE)
        print("done")
    except:
        print("oops")

if __name__ == "__main__":
    main()