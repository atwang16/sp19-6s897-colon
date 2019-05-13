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
        self.X_test_orig, self.X_test, self.y_test = self.load_images("test", format, preserve_original=True)
        self.format = format

    @staticmethod
    def normalize(mat):
        mean = np.mean(mat)
        std = np.std(mat)
        return (mat - mean) / std

    # assuming they all have the same (or similar names) and are alphabetical
    def load_images(self, split, seg_format, percent=1.0, preserve_original=False):
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
            if preserve_original:
                original_images = [None for _ in range(len(original_image_files))]
            else:
                original_images = None

            for i, image_path in enumerate(original_image_files):
                segmentation_path = segmentation_files[i]
                imgs = self.read_image(image_path, preserve_original)
                if preserve_original:
                    original_images[i], images[i, :, :, :] = imgs
                else:
                    images[i, :, :, :] = imgs
                labels[i, :] = self.read_segmentation(segmentation_path, seg_format)

            if preserve_original:
                return original_images, images, labels
            else:
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

    def read_image(self, path, preserve_original=False):
        raw_image = cv2.imread(path)
        rescaled_image = cv2.resize(raw_image, self.input_shape[:2], interpolation=cv2.INTER_CUBIC)
        if preserve_original:
            return rescaled_image, Dataset.normalize(np.array(rescaled_image))
        else:
            return Dataset.normalize(np.array(rescaled_image))

    @staticmethod
    def get_bounds(img):
        nonzeros = np.nonzero(img)
        y_min = nonzeros[0][0]
        y_max = nonzeros[0][-1]
        x_min = np.min(nonzeros[1])
        x_max = np.max(nonzeros[1])
        return x_min, y_min, x_max, y_max

    def read_segmentation(self, path, format):

        raw_segmentation = cv2.imread(path)
        rescaled_segmentation = cv2.resize(raw_segmentation, self.input_shape[:2], interpolation=cv2.INTER_CUBIC)

        x_min, y_min, x_max, y_max = self.get_bounds(rescaled_segmentation)

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

class YoloDataset(Dataset):
    def __init__(self, data_path, anchors, new_shape=(224,224)):
        self.anchors = anchors
        super().__init__(data_path, new_shape, LocFormat.BOX)

    @staticmethod
    def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
        '''Preprocess true boxes to training input format

        Parameters
        ----------
        true_boxes: array, shape=(m, T, 5)
            Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
        input_shape: array-like, hw, multiples of 32
        anchors: array, shape=(N, 2), wh
        num_classes: integer

        Returns
        -------
        y_true: list of array, shape like yolo_outputs, xywh are relative value

        '''
        assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
        num_layers = len(anchors) // 3  # default setting
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

        true_boxes = np.array(true_boxes, dtype='float32')
        input_shape = np.array(input_shape, dtype='int32')
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        m = true_boxes.shape[0]
        grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                           dtype='float32') for l in range(num_layers)]

        # Expand dim to apply broadcasting.
        anchors = np.expand_dims(anchors, 0)
        anchor_maxes = anchors / 2.
        anchor_mins = -anchor_maxes
        valid_mask = boxes_wh[..., 0] > 0

        for b in range(m):
            # Discard zero rows.
            wh = boxes_wh[b, valid_mask[b]]
            if len(wh) == 0: continue
            # Expand dim to apply broadcasting.
            wh = np.expand_dims(wh, -2)
            box_maxes = wh / 2.
            box_mins = -box_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)

            # Find best anchor for each true box
            best_anchor = np.argmax(iou, axis=-1)

            for t, n in enumerate(best_anchor):
                for l in range(num_layers):
                    if n in anchor_mask[l]:
                        i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                        k = anchor_mask[l].index(n)
                        c = true_boxes[b, t, 4].astype('int32')
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5 + c] = 1

        return y_true

    # assuming they all have the same (or similar names) and are alphabetical
    def load_images(self, split, seg_format, percent=1.0, preserve_original=False):
        original_image_files = sorted((os.path.join(self.images_dir[split], "polyps", f)
                                       for f in os.listdir(os.path.join(self.images_dir[split], "polyps"))
                                       if f[0] != "."), key=natural_keys)
        segmentation_files = sorted((os.path.join(self.images_dir[split], "segmentations", f)
                                     for f in os.listdir(os.path.join(self.images_dir[split], "segmentations"))
                                     if f[0] != "."), key=natural_keys)

        assert len(original_image_files) == len(segmentation_files), f"Error: found {len(original_image_files)} images but {len(segmentation_files)} segmentations"

        segmentation_shape = (5,)  # for class ID

        if percent == 0.0 or percent == 1.0:
            images = np.zeros((len(original_image_files), *self.input_shape))
            labels = np.zeros((len(original_image_files), 1, *segmentation_shape))

            for i, image_path in enumerate(original_image_files):
                segmentation_path = segmentation_files[i]
                images[i, :, :, :] = self.read_image(image_path)
                labels[i, :] = self.read_segmentation(segmentation_path, seg_format)

            y_true = YoloDataset.preprocess_true_boxes(labels, self.input_shape[:2], self.anchors, 1)
            return [images, *y_true], np.zeros((len(original_image_files), *segmentation_shape))
        else:
            split_1_size = int(len(original_image_files) * percent)
            split_2_size = len(original_image_files) - split_1_size
            images_split_1 = np.zeros((split_1_size, *self.input_shape))
            labels_split_1 = np.zeros((split_1_size, 1, *segmentation_shape))
            images_split_2 = np.zeros((split_2_size, *self.input_shape))
            labels_split_2 = np.zeros((split_2_size, 1, *segmentation_shape))

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

            y_true_split_1 = YoloDataset.preprocess_true_boxes(labels_split_1, self.input_shape[:2], self.anchors, 1)
            y_true_split_2 = YoloDataset.preprocess_true_boxes(labels_split_2, self.input_shape[:2], self.anchors, 1)
            return [images_split_1, *y_true_split_1], np.zeros((split_1_size, *segmentation_shape)), \
                   [images_split_2, *y_true_split_2], np.zeros((split_2_size, *segmentation_shape))

    def read_segmentation(self, path, format):
        raw_segmentation = cv2.imread(path)
        rescaled_segmentation = cv2.resize(raw_segmentation, self.input_shape[:2], interpolation=cv2.INTER_CUBIC)

        x_min, y_min, x_max, y_max = self.get_bounds(rescaled_segmentation)

        return np.array([x_min, y_min, x_max, y_max, 0])
