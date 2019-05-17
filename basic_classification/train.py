# importing model file
<<<<<<< HEAD
from models.vgg19 import vgg19
=======
from models.vgg19 import vgg19, vgg19_dropout
>>>>>>> old-working-version
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
# to run locally, run in the /basic_classification directory: python3 train.py --base_path . (then arguments)

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
parser.add_argument('--data_aug', type=float, default=False, help='Boolean for data augmentation')

# Model Hyper parameters
parser.add_argument('--model_type', type=str, default='vgg19', help='Name of the preinitialized model to use out of { resnet50 | vgg19 | vgg16 }')
parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs to train the model')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate to train the model')
parser.add_argument('--loss', type=str, default='sparse_categorical_crossentropy', help='Loss function to train the model with (kullback_leibler_divergence | binary_crossentropy)')

# model save/load parameters
parser.add_argument('--model_name', type=str, default='polyp_detection_model', help='Name of the file to save the model (do not put .h5)')
parser.add_argument('--load_model', type=str, default=None, help='Name of a model that will be loaded')

print('\n=== Setting up Parameters ===\n')

args = parser.parse_args()
args.train_path = '/'.join([args.base_path, args.train_path])
args.valid_path = '/'.join([args.base_path, args.valid_path])


print('\n=== Initiating Model ===\n')

K.set_image_data_format('channels_last')
if args.load_model is not None:
    model = load_model(args.load_model)
elif args.model_type == 'resnet50':
	model = resnet50((224, 224, 3), 2)
elif args.model_type == 'vgg19':
	model = vgg19((224, 224, 3), 2)
elif args.model_type == 'vgg19_dropout':
	model = vgg19_dropout((224, 224, 3), 2)
elif args.model_type == 'vgg16':
	model = vgg16((224, 224, 3), 2)


print('\n=== Compiling Model ===\n')

adam = optimizers.Adam(lr=args.lr)
model.compile(optimizer=adam, loss=args.loss, metrics=['accuracy'])


print('\n=== Training Model ===\n')

model = train_model_from_dir(args.base_path, args.train_path, args.valid_path, model, epochs=args.num_epochs)


#END FILE
