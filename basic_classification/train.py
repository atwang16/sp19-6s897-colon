# importing model file
from models.vgg19 import vgg19
from models.vgg16 import vgg16
from models.resnet50 import resnet50

# importing train file
from train_from_directory import train_model_from_dir

import keras
import tensorflow as tf

from keras import optimizers
from keras.models import load_model

import argparse
from keras import backend as K

import numpy as np

# to run on gcp, use: CUDA_VISIBLE_DEVICES=0 ./docker_run.sh python3 /basic_classification/train.py (then arguments)

# Loading arguments
parser = argparse.ArgumentParser(description='Polyp Detecting Model')

# evaluation
parser.add_argument('--only_test', type=bool, default=False, help='Flag such that, if true, the model will only be evaluated on the dataset passed in split by train_percent, validation_percent, etc')
parser.add_argument('--base_path', default="/basic_classification")

# data location
parser.add_argument('--train_path', type=str, default='data/kvasir_train_test_split/train', help='folder that contains all train images')
parser.add_argument('--valid_path', type=str, default='data/kvasir_train_test_split/val', help='folder that contains all validation images')
parser.add_argument('--model_type', type=str, default='vgg19', help='Name of a model to use out of { vgg19 | vgg16 | resnet50 }')

# data parameters
parser.add_argument('--validation_split', type=float, default=0.2, help='Percent of data to use for validation (rest will be used for training)')

# Model Hyper parameters
parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs to train the model')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate to train the model')
parser.add_argument('--loss', type=str, default='sparse_categorical_crossentropy', help='Loss function to train the model with (kullback_leibler_divergence | binary_crossentropy)')

# model save/load parameters
parser.add_argument('--model_name', type=str, default='polyp_detection_model', help='Name of the file to save the model (do not put .h5)')
parser.add_argument('--load_model', type=str, default=None, help='Name of a model that will be loaded')

print('\n=== Setting up Parameters ===\n')

args = parser.parse_args()
args.train_path = '/'.join([args.base_path, args.train_path])
args.valid_path = '/'.join([args.base_path, args.valid_path])

# print("VERIFY USING GPU")
# # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# gpu_devices = K.tensorflow_backend._get_available_gpus()
# print(f"GPU devices: {gpu_devices}")
# print("----------------")

print('\n=== Initiating Model ===\n')

if args.load_model is not None:
    model = load_model(args.load_model)
elif args.model_type == 'vgg19':
	K.set_image_data_format('channels_last')
	model = vgg19((224, 224, 3), 2)
elif args.model_type == 'vgg16':
	K.set_image_data_format('channels_last')
	model = vgg16((224, 224, 3), 2)
elif args.model_type == 'resnet50':
	K.set_image_data_format('channels_last')
	model = resnet50((224, 224, 3), 2)


print('\n=== Compiling Model ===\n')

adam = optimizers.Adam(lr=args.lr)
model.compile(optimizer=adam, loss=args.loss, metrics=['accuracy'])


print('\n=== Training Model ===\n')

train_model_from_dir(args.train_path, args.valid_path, model, model_name=args.model_type, epochs=args.num_epochs)


#END FILE
