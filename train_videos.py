#!/usr/bin/env python3

# importing model file
import models.vgg19 as vgg19

# importing dataset file
import dataset as data

from roc_callback import auc_metric

import keras
import tensorflow as tf
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import argparse
import os

import PIL

MODELS_DIRECTORY = os.path.join("experiments")  # TODO: fix this

if __name__ == '__main__':
    # Loading arguments
    parser = argparse.ArgumentParser(description='Polyp Detecting Model')

    # evaluation
    parser.add_argument('--only_test', type=bool, default=False,
                        help='Flag such that, if true, the model will only be evaluated on the dataset passed in split by train_percent, validation_percent, etc')

    # data location
    parser.add_argument('--dataset', type=str, default='ETIS-LaribPolypDB/GroundTruth/',
                        help='folder that contains all images that the model will be trained on')
    parser.add_argument('--ground_truth', type=str, default='ETIS-LaribPolypDB/ETIS-LaribPolypDB/',
                        help='folder that contains all images that contain the ground truth files')

    # data parameters
    parser.add_argument('--train_percent', type=float, default=0.2, help='Percent of data to use for training')
    parser.add_argument('--validation_percent', type=float, default=0.1, help='Percent of data to use for validation')

    # Model Hyper parameters
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate to train the model')
    parser.add_argument('--loss', type=str, default='binary_crossentropy',
                        help='Loss function to train the model with (binary_crossentropy | categorical_crossentropy)')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes to separate the data into')

    # model save/load parameters
    parser.add_argument('--model_name', type=str, default='polyp_detection_model',
                        help='Name of the file to save the model (do not put .h5)')
    parser.add_argument('--load_model', type=str, default=None, help='Name of a model that will be loaded')

    print('\n=== Setting up Parameters ===\n')

    args = parser.parse_args()

    # config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
    # sess = tf.Session(config=config)
    # keras.backend.set_session(sess)

    print('\n=== Setting Up Data ===\n')

    dataset = data.Dataset(data_path=data.VIDEO_DATASET, label_path=data.VIDEO_LABELS)

    print('\n=== Initiating Model ===\n')

    if args.load_model is not None:
        model = load_model(args.load_model)
    else:
        model = vgg19.vgg(dataset.input_shape, args.num_classes)

    print('\n=== Compiling Model ===\n')

    # optimizer
    adam = optimizers.Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # we expect to only separate between two different classes : polyp or not polyp
    model.compile(optimizer=adam, loss=args.loss, metrics=['binary_accuracy', 'categorical_accuracy', auc_metric])

    checkpoint = ModelCheckpoint(filepath=MODELS_DIRECTORY,
                                 monitor='val_abs_error',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')

    callbacks = [checkpoint]

    if args.only_test:

        print('\n=== Training Model ===\n')

        # training the model
        model.fit(dataset.get_X("train"), dataset.get_y("train"),
                  validation_data=(dataset.get_X("val"), dataset.get_y("val")),
                  epochs=args.num_epochs,
                  callbacks=callbacks)

        print('\n=== Saving Model ===\n')

        model.save(args.model_name + '.h5')

    # print('\n=== Evaluating Model ===\n')
    #
    # # evaluating model
    # score = model.evaluate(test_patches, test_labels)
    # print(score)

#END FILE
