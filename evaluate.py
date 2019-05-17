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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import cv2
from keras.losses import mean_squared_error

IMAGES_DIR = "data/segmentation"
NORMALIZE_LABELS = True
INPUT_SHAPE = (224, 224)


def get_localization_format(typ):
    if typ in {"yolov3", "baseline"}:
        return data.LocFormat.BOX
    elif typ in {"vgg19", "resnet50"}:
        return data.LocFormat.CENTER
    else:
        raise ValueError("Model type not supported.")


def get_model(typ, input_shape, pretrained_weights):
    if typ == 'vgg19':
        model = vgg.vgg19(input_shape, pretrained_weights=pretrained_weights, use_sigmoid=True)
        loss = "mean_squared_error"
    elif typ == 'resnet50':
        model = resnet.resnet50(input_shape, pretrained_weights=pretrained_weights, use_sigmoid=True)
        loss = "mean_squared_error"
    elif typ == "yolov3":
        model = yolo.yolov3(input_shape, pretrained_weights=pretrained_weights, freeze_body=2)
        loss = {'yolo_loss': lambda y_true, y_pred: y_pred}
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

        intersection_area = mx(0.0, intersection_x_max - intersection_x_min + 1e-10) * \
                            mx(0.0, intersection_y_max - intersection_y_min + 1e-10)
        true_area = (y_true[:, 2] - y_true[:, 0] + 1e-10) * (y_true[:, 3] - y_true[:, 1] + 1e-10)
        pred_area = (y_pred[:, 2] - y_pred[:, 0] + 1e-10) * (y_pred[:, 3] - y_pred[:, 1] + 1e-10)
        total_area = true_area + pred_area

        return 2 * mean(intersection_area / total_area)


    def dice_score_center(y_true, y_pred, is_tf_metric=True):
        if is_tf_metric:
            st = tf.stack
            sq = K.square
        else:
            st = np.stack
            sq = np.square

        width_height = sq(y_pred[:, 2:4])

        y_true_box = st([y_true[:, 0] - y_true[:, 2] / 2, y_true[:, 1] - y_true[:, 3] / 2,
                               y_true[:, 0] + y_true[:, 2] / 2, y_true[:, 1] + y_true[:, 3] / 2], 1)
        y_pred_box = st([y_pred[:, 0] - width_height[:, 0] / 2, y_pred[:, 1] - width_height[:, 1] / 2,
                               y_pred[:, 0] + width_height[:, 0] / 2, y_pred[:, 1] + width_height[:, 1] / 2], 1)

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


def dice_score_box_loss(y_true, y_pred):
    return 1 - get_dice_score(LocFormat.BOX)(y_true, y_pred, is_tf_metric=True)


def mse_dice_loss(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    # pred_area = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])
    # large_bb = K.mean(K.square(pred_area))
    dice_loss = dice_score_box_loss(y_true, y_pred)
    return mse + 0.5 * dice_loss


def mse_sqrt_loss(y_true, y_pred):
    mse_center = mean_squared_error(y_true[:, 0:2], y_pred[:, 0:2])
    mse_size = mean_squared_error(y_true[:, 2:4], K.square(y_pred[:, 2:4]))
    # pred_area = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])
    # large_bb = K.mean(K.square(pred_area))
    return mse_center + mse_size


def evaluate(model, dataset, split, typ):
    X = dataset.__getattribute__(f"X_{split}")
    y = dataset.__getattribute__(f"y_{split}")

    if typ == "yolov3":
        y_pred = model.evaluate_on_image(X)
    elif typ == "baseline":
        y_pred = np.array([[0, 0, dataset.input_shape[0], dataset.input_shape[1]] for _ in range(y.shape[0])])
    else:
        y_pred = model.predict(X)
    assert y.shape == y_pred.shape
    print(y_pred)

    score = get_dice_score(dataset.format)(y, y_pred, is_tf_metric=False)
    distr = [get_dice_score(dataset.format)(y[np.newaxis, i, :], y_pred[np.newaxis, i, :], is_tf_metric=False) for i in range(y.shape[0])]
    return y_pred, score, distr


def visualize(save_dir, dataset, predictions, num_to_generate=None):
    def get_bounding_box(box, color, linewidth=1):
        y_min, x_min, y_max, x_max = box
        if NORMALIZE_LABELS:
            x_min *= INPUT_SHAPE[0]
            x_max *= INPUT_SHAPE[0]
            y_min *= INPUT_SHAPE[1]
            y_max *= INPUT_SHAPE[1]
        return patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=linewidth, edgecolor=color,
                                 facecolor='none')

    def get_bounding_box_center(box, color, linewidth=1):
        x_center, y_center, width, height = box
        width *= width
        height *= height
        if NORMALIZE_LABELS:
            x_center *= INPUT_SHAPE[0]
            width *= INPUT_SHAPE[0]
            y_center *= INPUT_SHAPE[1]
            height *= INPUT_SHAPE[1]
        return patches.Rectangle((x_center - width / 2, y_center - height / 2), width, height, linewidth=linewidth, edgecolor=color, facecolor='none')

    X = dataset.__getattribute__("X_test_orig")
    y = dataset.__getattribute__("y_test")
    gbb = get_bounding_box if dataset.format == LocFormat.BOX else get_bounding_box_center

    if num_to_generate is not None and num_to_generate < len(X):
        X = X[:num_to_generate]
        y = y[:num_to_generate, ...]

    for i in range(len(X)):
        img = X[i]

        # Create figure and axes
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Create a Rectangle patch
        true_bb_rep = y[i, ...]
        true_box = gbb(true_bb_rep, color="r", linewidth=3)
        # for j in range(len(predictions[i])):
        #     pred_box = gbb(predictions[i], color=(random.uniform(0, 1), random.uniform(0, 1), 0))
        pred_box = gbb(predictions[i], color="b")
        ax.add_patch(pred_box)

        # Add the patch to the Axes
        ax.add_patch(true_box)
        plt.axis('off')
        fig.savefig(os.path.join(save_dir, f"test_polyp_{i}.png"), bbox_inches=0)
        plt.close(fig)


if __name__ == '__main__':
    # Loading arguments
    parser = argparse.ArgumentParser(description='Polyp Detecting Model')

    # Model hyperparameters
    parser.add_argument('--type', type=str, default='vgg19',
                        help='Determines which convolutional model to use. Valid options are {vgg19|resnet50|pvgg19}')
    parser.add_argument('--load_model', type=str, help='Name of a model that will be loaded')
    parser.add_argument('--save_dir', type=str, help='Directory in which to save visualizations', required=True)
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

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print('\n=== Setting Up Data ===\n')

    dataset = data.Dataset(IMAGES_DIR, format=get_localization_format(args.type), normalize_labels=NORMALIZE_LABELS)

    print('\n=== Initiating Model ===\n')
    if args.type != "baseline":
        model, loss_function = get_model(args.type, dataset.input_shape, pretrained_weights=args.load_model)
        localization_format = get_localization_format(args.type)

        if args.type != "yolov3":
            model.summary()

        print('\n=== Compiling Model ===\n')

        # optimizer
        adam = optimizers.Adam(lr=0.0)  # beta_1=0.9, beta_2=0.999, decay=0.0

        if args.type != "yolov3":
            model.compile(optimizer=adam, loss=loss_function)
    else:
        model = None

    print('\n=== Evaluating Model ===\n')
    val_predictions, val_dice_score, val_dist = evaluate(model, dataset, "val", args.type)
    test_predictions, test_dice_score, test_dist = evaluate(model, dataset, "test", args.type)

    print(f"Val dice score = {val_dice_score}")
    print(f"Test dice score = {test_dice_score}")

    fig, ax = plt.subplots(1)

    # Display the image
    bins = np.linspace(0.0, 1.0, 25)
    ax.hist([val_dist, test_dist], bins, label=["val_dice_score", "test_dice_score"])
    ax.legend(loc='upper right')
    fig.savefig(os.path.join(args.save_dir, f"dice_score_dist.png"), bbox_inches=0)
    plt.close(fig)

    print("\n=== Visualizing Model ===\n")
    visualize(args.save_dir, dataset, test_predictions, num_to_generate=args.visualize)
