from config import *
from data import load_split

def main():
    print("start")
    x_train, y_train = load_split('train')
    x_val, y_val = load_split('val')
    x_test, y_test = load_split('test')
    print("done")

if __name__ == "__main__":
    main()