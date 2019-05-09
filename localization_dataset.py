import cv2
import numpy as np
import os
from enum import Enum
import re

def natural_keys(text):
    """
    Key for natural sorting. Returns list representation of string with letters and numbers separated.
    :param text: string representing value to be sorted.
    :return: list representing text, with strings of letters and numbers as separated tokens
    """
    def atoi(text):
        return int(text) if text.isdigit() else text

    return [atoi(c) for c in re.split('(\d+)', text)]

class LocFormat(Enum):
    BOX = 0
    CENTER = 1
    SEGMENTATION = 2

class Dataset:
    def __init__(self, data_path, new_shape=(224,224), format=LocFormat.BOX):
        self.input_shape = (*new_shape, 3)
        self.images_dir = {
            "train": os.path.join(data_path, "train"),
            "test": os.path.join(data_path, "test")
        }
        self.train_val_split = 0.75

        self.X_train, self.y_train, self.X_val, self.y_val = self.load_images("train", format, percent=self.train_val_split)
        self.X_test, self.y_test = self.load_images("test", format)

    @staticmethod
    def normalize(mat):
        mean = np.mean(mat)
        std = np.std(mat)
        return (mat - mean) / std

    # assuming they all have the same (or similar names) and are alphabetical
    def load_images(self, split, seg_format, percent=1.0):
        original_image_files = sorted((os.path.join(self.images_dir[split], "polyps", f)
                                       for f in os.listdir(os.path.join(self.images_dir[split], "polyps"))
                                       if f[0] != "."), key=natural_keys)
        segmentation_files = sorted((os.path.join(self.images_dir[split], "segmentations", f)
                                     for f in os.listdir(os.path.join(self.images_dir[split], "segmentations"))
                                     if f[0] != "."), key=natural_keys)

        assert len(original_image_files) == len(segmentation_files), f"Error: found {len(original_image_files)} images but {len(segmentation_files)} segmentations"

        segmentation_shape = self.input_shape if seg_format == LocFormat.SEGMENTATION else (4,)

        if percent == 0.0 or percent == 1.0:
            images = np.zeros((len(original_image_files), *self.input_shape))
            labels = np.zeros((len(original_image_files), *segmentation_shape))

            for i, image_path in enumerate(original_image_files):
                segmentation_path = segmentation_files[i]
                images[i, :, :, :] = self.read_image(image_path)
                labels[i, :] = self.read_segmentation(segmentation_path, seg_format)

            return images, labels
        else:
            split_1_size = int(len(original_image_files) * percent)
            split_2_size = len(original_image_files) - split_1_size
            images_split_1 = np.zeros((split_1_size, *self.input_shape))
            labels_split_1 = np.zeros((split_1_size, *segmentation_shape))
            images_split_2 = np.zeros((split_2_size, *self.input_shape))
            labels_split_2 = np.zeros((split_2_size, *segmentation_shape))

            i = 0
            while i < split_1_size:
                images_split_1[i, :, :, :] = self.read_image(original_image_files[i])
                labels_split_1[i, :] = self.read_segmentation(segmentation_files[i], seg_format)
                i += 1

            j = 0
            while j < split_2_size:
                images_split_2[j, :, :, :] = self.read_image(original_image_files[split_1_size+j])
                labels_split_2[j, :] = self.read_segmentation(segmentation_files[split_1_size+j], seg_format)
                j += 1

            return images_split_1, labels_split_1, images_split_2, labels_split_2

    def read_image(self, path):
        raw_image = cv2.imread(path)
        rescaled_image = cv2.resize(raw_image, self.input_shape[:2], interpolation=cv2.INTER_CUBIC)
        return Dataset.normalize(np.array(rescaled_image))

    def read_segmentation(self, path, format):
        def get_bounds(img):
            nonzeros = np.nonzero(img)
            x_min = nonzeros[0][0]
            x_max = nonzeros[0][-1]
            y_min = np.min(nonzeros[1])
            y_max = np.max(nonzeros[1])
            return x_min, y_min, x_max, y_max

        raw_segmentation = cv2.imread(path)
        rescaled_segmentation = cv2.resize(raw_segmentation, self.input_shape[:2], interpolation=cv2.INTER_CUBIC)

        x_min, y_min, x_max, y_max = get_bounds(rescaled_segmentation)

        if format == LocFormat.BOX:
            return np.array([x_min, y_min, x_max, y_max])
        elif format == LocFormat.CENTER:
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            return np.array([x_center, y_center, width, height])
        elif format == LocFormat.SEGMENTATION:
            return np.array(rescaled_segmentation)
        else:
            raise ValueError("Format not supported.")