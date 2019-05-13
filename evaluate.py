#!/usr/bin/env python3

from localization_dataset import LocFormat
import keras
import tensorflow as tf
from keras import backend as K
import argparse
import localization_dataset as data
import models.vgg19_localization as vgg
import models.resnet50_localization as resnet
import models.yolov3 as yolo
import numpy as np
from keras import optimizers

IMAGES_DIR = "data/segmentation"


def get_localization_format(typ):
    if typ in {"yolov3"}:
        return data.LocFormat.BOX
    elif typ in {"vgg19", "resnet50"}:
        return data.LocFormat.CENTER
    else:
        raise ValueError("Model type not supported.")


def get_model(typ, input_shape, pretrained_weights):
    if typ == 'vgg19':
        model = vgg.vgg19(input_shape, use_sigmoid=True)
        loss = "mean_squared_error"
    elif typ == 'resnet50':
        model = resnet.resnet50(input_shape)
        loss = "mean_squared_error"
    elif typ == "yolov3":
        model = yolo.YOLO(model_image_size=input_shape, model_path=pretrained_weights)
        loss = None
    else:
        raise ValueError(f"Model \"{typ}\" not supported.")
    return model, loss


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true), axis=-1))


def get_dice_score(fmt):
    def dice_score_box(y_true, y_pred, is_tf_metric=True):
        if is_tf_metric:
            mx = tf.math.maximum
            mn = tf.math.minimum
            mean = tf.math.reduce_mean
        else:
            mx = np.maximum
            mn = np.minimum
            mean = np.mean
        intersection_x_min = mx(y_true[:, 0], y_pred[:, 0])
        intersection_y_min = mx(y_true[:, 1], y_pred[:, 1])
        intersection_x_max = mn(y_true[:, 2], y_pred[:, 2])
        intersection_y_max = mn(y_true[:, 3], y_pred[:, 3])

        intersection_area = mx(0.0, intersection_x_max - intersection_x_min + 1) * \
                            mx(0.0, intersection_y_max - intersection_y_min + 1)
        true_area = (y_true[:, 2] - y_true[:, 0] + 1) * (y_true[:, 3] - y_true[:, 1] + 1)
        pred_area = (y_pred[:, 2] - y_pred[:, 0] + 1) * (y_pred[:, 3] - y_pred[:, 1] + 1)
        total_area = true_area + pred_area

        return 2 * mean(intersection_area / total_area)


    def dice_score_center(y_true, y_pred, is_tf_metric=True):
        if is_tf_metric:
            st = tf.stack
        else:
            st = np.stack
        y_true_box = st([y_true[:, 0] - y_true[:, 2] / 2, y_true[:, 1] - y_true[:, 3] / 2,
                               y_true[:, 0] + y_true[:, 2] / 2, y_true[:, 1] + y_true[:, 3] / 2], 1)
        y_pred_box = st([y_pred[:, 0] - y_pred[:, 2] / 2, y_pred[:, 1] - y_pred[:, 3] / 2,
                               y_pred[:, 0] + y_pred[:, 2] / 2, y_pred[:, 1] + y_pred[:, 3] / 2], 1)

        return dice_score_box(y_true_box, y_pred_box, is_tf_metric=is_tf_metric)

    def dice_score_seg(y_true, y_pred, is_tf_metric=True):
        raise NotImplementedError("Dice score for segmentation not yet implemented.")

    if fmt == LocFormat.BOX:
        return dice_score_box
    elif fmt == LocFormat.CENTER:
        return dice_score_center
    elif fmt == LocFormat.SEGMENTATION:
        return dice_score_seg
    else:
        raise ValueError(f"Format {fmt} not supported.")


def evaluate(model, dataset, split, typ):
    X = dataset.__getattribute__(f"X_{split}")
    y = dataset.__getattribute__(f"y_{split}")

    if typ == "yolov3":
        y_pred = model.evaluate_on_image(X)
    else:
        y_pred = model.predict(X)
    assert y.shape == y_pred.shape

    score = get_dice_score(dataset.format)(y, y_pred, is_tf_metric=False)
    return y_pred, score


def visualize(dataset, split, predictions, num_to_generate=None):
    pass


if __name__ == '__main__':
    # Loading arguments
    parser = argparse.ArgumentParser(description='Polyp Detecting Model')

    # Model hyperparameters
    parser.add_argument('--type', type=str, default='vgg19',
                        help='Determines which convolutional model to use. Valid options are {vgg19|resnet50|pvgg19}')
    parser.add_argument('--load_model', type=str, help='Name of a model that will be loaded', required=True)
    parser.add_argument('--visualize', type=int, help='Number of images to visualize', default=None)

    print(f"keras version: {keras.__version__}")
    print(f"tensorflow version: {tf.__version__}")

    print("VERIFY USING GPU")
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    gpu_devices = K.tensorflow_backend._get_available_gpus()
    print(f"GPU devices: {gpu_devices}")
    print("----------------")

    print('\n=== Setting up Parameters ===\n')

    args = parser.parse_args()

    print('\n=== Setting Up Data ===\n')

    dataset = data.Dataset(IMAGES_DIR, format=get_localization_format(args.type))

    print('\n=== Initiating Model ===\n')

    model, loss_function = get_model(args.type, dataset.input_shape, pretrained_weights=args.load_model)
    localization_format = get_localization_format(args.type)

    if args.type != "yolov3":
        model.summary()

    print('\n=== Compiling Model ===\n')

    # optimizer
    adam = optimizers.Adam(lr=0.0)  # beta_1=0.9, beta_2=0.999, decay=0.0

    if args.type != "yolov3":
        model.compile(optimizer=adam, loss=loss_function)

    print('\n=== Evaluating Model ===\n')
    val_predictions, val_dice_score = evaluate(model, dataset, "val", args.type)
    test_predictions, test_dice_score = evaluate(model, dataset, "test", args.type)

    print(f"Val dice score = {val_dice_score}")
    print(f"Test dice score = {test_dice_score}")

    print("\n=== Visualizing Model ===\n")
    visualize(dataset, "test", test_predictions, num_to_generate=args.visualize)
