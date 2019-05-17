
import os
import random
import math
import shutil
import argparse

DATA_DIR = 'data/kvasir'
DEST_DIR = 'data/kvasir_train_test_split'
TRAIN_VAL_TEST_RATIO = (0.8, 0.1, 0.1)
print(DATA_DIR)

def copy_files_into_split(data_dir=DATA_DIR, dest_dir=DEST_DIR, ratio=TRAIN_VAL_TEST_RATIO):
    for data_class in os.listdir(DATA_DIR):
        print(data_class)
        if os.path.isdir(os.path.join(DATA_DIR, data_class)):
            imgs = [img for img in os.listdir(os.path.join(DATA_DIR, data_class))]
            random.shuffle(imgs)
            val_idx = math.floor(TRAIN_VAL_TEST_RATIO[0] * len(imgs))
            test_idx = math.floor(TRAIN_VAL_TEST_RATIO[1] * len(imgs) + val_idx)
            print(len(imgs))
            for idx in range(len(imgs)):
                if idx < val_idx:
                    dataset = 'train'
                elif idx < test_idx:
                    dataset = 'val'
                else:
                    dataset = 'test'   
                if not os.path.exists(os.path.join(DEST_DIR, dataset, data_class)):
                    os.makedirs(os.path.join(DEST_DIR, dataset, data_class))
                shutil.copy(os.path.join(DATA_DIR, data_class, imgs[idx]), os.path.join(DEST_DIR, dataset, data_class, imgs[idx]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', action="store", default=DATA_DIR, dest='data_dir', help='Data directory')
    parser.add_argument('--destination', action="store", default=DEST_DIR, dest='destination', help='Destination directory for the copy')
    parser.add_argument('--train_test_ratio', action="store", default=TRAIN_VAL_TEST_RATIO, dest='ratio', help='Train test split ratio')

    args = parser.parse_args()

    copy_files_into_split(data_dir=args.data_dir, dest_dir=args.destination, ratio=args.ratio)