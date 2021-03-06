#!/usr/bin/env python3

import localization_dataset as data
import evaluate
import models.vgg19_localization as vgg
import models.resnet50_localization as resnet
import models.yolov3 as yolo

from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
import argparse
import numpy as np
import keras
import tensorflow as tf
from keras import backend as K

import matplotlib.pyplot as plt
import os
import time

IMAGES_DIR = "data/segmentation"

def get_localization_format(typ):
    if typ in {"yolov3"}:
        return data.LocFormat.BOX
    elif typ in {"vgg19", "resnet50"}:
        return data.LocFormat.CENTER
    else:
        raise ValueError("Model type not supported.")


def abs_error(y_true, y_pred):
    y_true = K.sum(y_true)
    y_pred = K.sum(y_pred)
    return K.abs(y_true-y_pred)


def get_model(typ, input_shape, pretrained_weights):
    if typ == 'vgg19':
        model = vgg.vgg19(input_shape, pretrained_weights=pretrained_weights, use_sigmoid=True)
        loss = "mean_squared_error"
    elif typ == 'resnet50':
        model = resnet.resnet50(input_shape, pretrained_weights=pretrained_weights, use_sigmoid=True)
        loss = evaluate.mse_sqrt_loss
    elif typ == "yolov3":
        model = yolo.yolov3(input_shape, pretrained_weights=pretrained_weights, freeze_body=2)
        loss = {'yolo_loss': lambda y_true, y_pred: y_pred}
    else:
        raise ValueError(f"Model \"{typ}\" not supported.")
    return model, loss


if __name__ == '__main__':
    # Loading arguments
    parser = argparse.ArgumentParser(description='Polyp Detecting Model')

    # Model hyperparameters
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--num_frozen', type=int, default=10, help='Number of frozen epochs to train the model')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate to train the model')
    parser.add_argument('--batch_size', type=int, default=8, help='Learning rate to train the model')
    parser.add_argument('--type', type=str, default='vgg19', help='Determines which convolutional model to use. Valid options are {vgg19|resnet50|pvgg19}')

    # model save/load parameters
    parser.add_argument('--model_name', type=str, help='Name of the file to save the model (do not put .h5)')
    parser.add_argument('--load_model', type=str, default=None, help='Name of a model that will be loaded')
    parser.add_argument('--output_dir', type=str, default='output/', help='Location to store outputs from the training.')

    print(f"keras version: {keras.__version__}")
    print(f"tensorflow version: {tf.__version__}")

    print("VERIFY USING GPU")
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    gpu_devices = K.tensorflow_backend._get_available_gpus()
    print(f"GPU devices: {gpu_devices}")
    print("----------------")

    print('\n=== Setting up Parameters ===\n')

    args = parser.parse_args()
    if args.model_name is not None:
        model_name = args.model_name
    else:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        model_name = f"{timestamp}_{args.type}"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
    # sess = tf.Session(config=config)
    # keras.backend.set_session(sess)

    print('\n=== Setting Up Data ===\n')

    if args.type == "yolov3":
        dataset = data.YoloDataset(IMAGES_DIR, yolo.get_anchors(yolo.ANCHOR_PATH))
    else:  # resnet, vgg
        dataset = data.Dataset(IMAGES_DIR, format=get_localization_format(args.type), normalize_labels=True)

    print('\n=== Initiating Model ===\n')

    model, loss_function = get_model(args.type, dataset.input_shape, pretrained_weights=args.load_model)
    localization_format = get_localization_format(args.type)

    model.summary()

    print('\n=== Compiling Model ===\n')

    # Prepare callbacks for model saving and for learning rate adjustment.
    logging = TensorBoard(log_dir=args.output_dir)
    checkpoint_name = '%s_model.{epoch:03d}.h5' % args.type
    checkpoint = ModelCheckpoint(filepath=os.path.join(args.output_dir, checkpoint_name),
                                 monitor='val_abs_error' if args.type == "yolov3" else "val_dice_score_center",
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='min' if args.type == "yolov3" else 'max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1**0.5, patience=12, verbose=1)
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # optimizer
    adam = optimizers.Adam(lr=args.lr)  # beta_1=0.9, beta_2=0.999, decay=0.0

    if args.type == "yolov3":
        model.compile(optimizer=adam, loss=loss_function, metrics=[abs_error])
    else:
        model.compile(optimizer=adam, loss=loss_function, metrics=[evaluate.rmse, evaluate.get_dice_score(localization_format), abs_error])

    print('\n=== Training Model ===\n')

    early_stop = True
    try:
        print ("= Frozen Training =")

        initial_epoch = 0
        epochs_to_train = max(1, min(args.num_frozen, args.num_epochs // 2))
        model.fit(dataset.X_train, dataset.y_train,
                  validation_data=(dataset.X_val, dataset.y_val),
                  epochs=epochs_to_train,
                  initial_epoch=initial_epoch,
                  batch_size=args.batch_size,
                  callbacks=[logging, checkpoint],
                  shuffle=True)
        model.save_weights(os.path.join(args.output_dir, f"{model_name}_initial.h5"))
        initial_epoch += epochs_to_train

        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        if args.type == "yolov3":
            model.compile(optimizer=adam, loss=loss_function, metrics=[abs_error])
        else:
            model.compile(optimizer=adam, loss=loss_function,
                          metrics=[evaluate.rmse, evaluate.get_dice_score(localization_format), abs_error])
        # train model
        checkpoint = ModelCheckpoint(filepath=os.path.join(args.output_dir, checkpoint_name),
                                     monitor='val_abs_error' if args.type == "yolov3" else "val_dice_score_center",
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='min' if args.type == "yolov3" else 'max')
        print("= Regular Training =")
        model.fit(dataset.X_train, dataset.y_train,
                  validation_data=(dataset.X_val, dataset.y_val),
                  epochs=args.num_epochs,
                  initial_epoch=initial_epoch,
                  batch_size=args.batch_size,
                  callbacks=[logging, checkpoint, reduce_lr],
                  shuffle=True)

        print('\n=== Saving Model ===\n')
        model.save_weights(os.path.join(args.output_dir, f"{model_name}_final.h5"))

        early_stop = False

    finally:
        if early_stop:
            model.save(os.path.join(args.output_dir, f'{model_name}_early_stop.h5'))
