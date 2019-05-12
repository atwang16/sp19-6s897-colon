# importing model file
from models.vgg19 import vgg

# importing train file
from train_from_directory import train_model_from_dir

import keras
import tensorflow as tf
from keras import optimizers
from keras.models import load_model
import argparse

import numpy as np

# Loading arguments
parser = argparse.ArgumentParser(description='Polyp Detecting Model')

# evaluation
parser.add_argument('--only_test', type=bool, default=False, help='Flag such that, if true, the model will only be evaluated on the dataset passed in split by train_percent, validation_percent, etc')

# data location
parser.add_argument('--train_path', type=str, default='data/kvasir_train_test_split/train', help='folder that contains all train images')
parser.add_argument('--valid_path', type=str, default='data/kvasir_train_test_split/val', help='folder that contains all validation images')

# data parameters
parser.add_argument('--validation_split', type=float, default=0.2, help='Percent of data to use for validation (rest will be used for training)')

# Model Hyper parameters
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train the model')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate to train the model')
parser.add_argument('--loss', type=str, default='sparse_categorical_crossentropy', help='Loss function to train the model with (binary_crossentropy | categorical_crossentropy)')

# model save/load parameters
parser.add_argument('--model_name', type=str, default='polyp_detection_model', help='Name of the file to save the model (do not put .h5)')
parser.add_argument('--load_model', type=str, default=None, help='Name of a model that will be loaded')

print('\n=== Setting up Parameters ===\n')

args = parser.parse_args()

# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)


print('\n=== Initiating Model ===\n')

if args.load_model is not None:
    model = load_model(args.load_model)
else:
    model = vgg((224, 224, 3), 2) # input size = (224, 224), number of classes = 2

print('\n=== Compiling Model ===\n')

# optimizer
adam = optimizers.Adam(lr=args.lr)

# we expect to only separate between two different classes : polyp or not polyp
model.compile(optimizer=adam, loss=args.loss, metrics=['binary_accuracy', 'categorical_accuracy'])


print('\n=== Training Model ===\n')

train_model_from_dir(args.train_path, args.valid_path, model, epochs=args.num_epochs)




#END FILE
