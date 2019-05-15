# importing model file
import models.vgg19 as vgg19
import models.patch_vgg19 as pvgg19
import models.resnet50 as resnet50

# importing dataset file
import patch_dataset as data

from roc_callback import auc_metric

import keras
import tensorflow as tf
from keras import optimizers
from keras.models import load_model
import argparse

import numpy as np
import matplotlib.pyplot as plt
import os

import keras.losses
import sklearn.utils as sk

keras.losses.custom_loss = auc_metric

# Loading arguments
parser = argparse.ArgumentParser(description='Polyp Detecting Model')

# evaluation
parser.add_argument('--only_test', type=bool, default=False, help='Flag such that, if true, the model will only be evaluated on the dataset passed in split by train_percent, validation_percent, etc')

# data location
parser.add_argument('--ground_truth', type=str, default='ETIS-LaribPolypDB/GroundTruth/', help='folder that contains all images that the model will be trained on')
parser.add_argument('--training_images', type=str, default='ETIS-LaribPolypDB/ETIS-LaribPolypDB/', help='folder that contains all images that contain the ground truth files')

# data parameters
parser.add_argument('--random_patches', type=bool, default=True, help='Boolean parameter, True means we sample patches randomly from the dataset images and False is the opposite')
parser.add_argument('--train_percent', type=float, default=0.2, help='Percent of data to use for training')
parser.add_argument('--validation_percent', type=float, default=0.1, help='Percent of data to use for validation')

# Model Hyper parameters
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train the model')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate to train the model')
parser.add_argument('--loss', type=str, default='categorical_crossentropy', help='Loss function to train the model with (binary_crossentropy | categorical_crossentropy)')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes to separate the data into')
parser.add_argument('--patch_size', type=int, default=32, help='Number of pixels per side in the patch')
parser.add_argument('--num_patches', type=int, default=1, help='Number of patches generated per image')
parser.add_argument('--resize_imgs', type=bool, default=True, help='Flag that will resize the images to (224,224) if True')
parser.add_argument('--type', type=str, default='vgg19', help='Determines which convolutional model to use. Valid options are {vgg19|resnet50|pvgg19}')

# model save/load parameters
parser.add_argument('--model_name', type=str, default='polyp_detection_model', help='Name of the file to save the model (do not put .h5)')
parser.add_argument('--load_model', type=str, default=None, help='Name of a model that will be loaded')
parser.add_argument('--output_dir', type=str, default='output/', help='Location to store outputs from the training.')

print('\n=== Setting up Parameters ===\n')

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

print("VERIFY USING GPU")
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
gpu_devices = K.tensorflow_backend._get_available_gpus()
print("GPU devices: ",gpu_devices)
print("----------------")
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)

print('\n=== Setting Up Data ===\n')

print('=== Training Data ===')
training_generator = data.Generator_Dataset_Rotated_Regression(args.patch_size, args.training_images, args.ground_truth, batch_size=args.batch_size)

print('\n=== Initiating Model ===\n')

if args.load_model is not None:
    model = load_model(args.load_model, custom_objects={'auc_metric': auc_metric})
else:
    if args.type == 'vgg19':
        model = vgg19.vgg(dataset.input_shape, args.num_classes)
    elif args.type == 'resnet50':
        model = resnet50.resnet(dataset.input_shape, args.num_classes)
    elif args.type == 'pvgg19':
        model = pvgg19.patch_vgg_regress(dataset.input_shape, args.num_classes)
        import pdb; pdb.set_trace()

model.summary()

print('\n=== Compiling Model ===\n')

# optimizer
adam = optimizers.Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# we expect to only separate between two different classes : polyp or not polyp
model.compile(optimizer=adam, loss='mean_squared_error')

early_stop = True
try:
    if not args.only_test:
        print('\n=== Training Model ===\n')
        # training the model
        model.fit_generator(generator=training_generator,
                            steps_per_epoch=len(training_generator), epochs=args.num_epochs)

        print('\n=== Saving Model ===\n')

        model.save(args.output_dir+'/'+args.model_name+'.h5')

    print('\n=== Testing Model ===\n')
    # evaluating model
    # loss, binary_accuracy, categorical_accuracy, auc

    early_stop = False

finally:
    print('\n=== Saving Model ===')
    if early_stop and not args.only_test:
        model.save(args.output_dir+args.model_name+'_early_stop.h5')










#END FILE
