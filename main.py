import os
from config import *

def main():
    print("start")
    if not os.path.exists(DATA_DIR):
        print("no file")

if __name__ == "__main__":
    main()