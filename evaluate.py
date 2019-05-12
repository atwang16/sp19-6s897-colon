#!/usr/bin/env python3

from localization_dataset import LocFormat
import tensorflow as tf


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true), axis=-1))


def get_dice_score(fmt):
    def dice_score_box(y_true, y_pred):
        intersection_x_min = tf.math.maximum(y_true[:, 0], y_pred[:, 0])
        intersection_y_min = tf.math.maximum(y_true[:, 1], y_pred[:, 1])
        intersection_x_max = tf.math.minimum(y_true[:, 2], y_pred[:, 2])
        intersection_y_max = tf.math.minimum(y_true[:, 3], y_pred[:, 3])

        intersection_area = tf.math.maximum(0.0, intersection_x_max - intersection_x_min + 1) * \
                            tf.math.maximum(0.0, intersection_y_max - intersection_y_min + 1)
        true_area = (y_true[:, 2] - y_true[:, 0] + 1) * (y_true[:, 3] - y_true[:, 1] + 1)
        pred_area = (y_pred[:, 2] - y_pred[:, 0] + 1) * (y_pred[:, 3] - y_pred[:, 1] + 1)
        total_area = true_area + pred_area

        return 2 * tf.math.reduce_mean(intersection_area / total_area)


    def dice_score_center(y_true, y_pred):
        y_true_box = tf.stack([y_true[:, 0] - y_true[:, 2] / 2, y_true[:, 1] - y_true[:, 3] / 2,
                               y_true[:, 0] + y_true[:, 2] / 2, y_true[:, 1] + y_true[:, 3] / 2], 1)
        y_pred_box = tf.stack([y_pred[:, 0] - y_pred[:, 2] / 2, y_pred[:, 1] - y_pred[:, 3] / 2,
                               y_pred[:, 0] + y_pred[:, 2] / 2, y_pred[:, 1] + y_pred[:, 3] / 2], 1)

        return dice_score_box(y_true_box, y_pred_box)

    def dice_score_seg(y_true, y_pred):
        raise NotImplementedError("Dice score for segmentation not yet implemented.")

    if fmt == LocFormat.BOX:
        return dice_score_box
    elif fmt == LocFormat.CENTER:
        return dice_score_center
    elif fmt == LocFormat.SEGMENTATION:
        return dice_score_seg
    else:
        raise ValueError(f"Format {fmt} not supported.")


# # evaluating model
# # loss, binary_accuracy, categorical_accuracy, auc
# loss, binary_accuracy, categorical_accuracy, auc = model.evaluate(dataset.X_test, dataset.y_test)
# print('TEST LOSS',loss)
# print('TEST BINARY ACCURACY',binary_accuracy)
# print('TEST CATEGORICAL ACCURACY',categorical_accuracy)
# print('TEST AUC',auc)