import sklearn.metrics as skm
import numpy as np

import patch_dataset as data
from keras.models import load_model

from roc_callback import auc_metric

import argparse

import os

from keras import backend as K

import tensorflow as tf


#python3 patch_majority_voting.py --images data/segmentation/test/polyps/ --ground_truth data/segmentation/test/segmentations/ --load_model

parser = argparse.ArgumentParser(description='Polyp Detecting Model Evalutaion')
# data location
parser.add_argument('--images', type=str, default='ETIS-LaribPolypDB/ETIS-LaribPolypDB/', help='folder that contains all images that the model will be trained on')
parser.add_argument('--ground_truth', type=str, default='ETIS-LaribPolypDB/GroundTruth/', help='folder that contains all images that contain the ground truth files')

parser.add_argument('--patch_size', type=int, default=32, help='Number of pixels per side in the patch')
parser.add_argument('--batch_size', type=int, default=1, help='Number of pixels per side in the patch')

parser.add_argument('--load_model', type=str, default=None, help='Name of a model that will be loaded')

gpu_devices = K.tensorflow_backend._get_available_gpus()
print("GPU devices: ",gpu_devices)
print("----------------")
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

args = parser.parse_args()

dataset = data.Dataset(args.patch_size, args.images, args.ground_truth)

model = load_model(args.load_model, custom_objects={'auc_metric': auc_metric})

ground_truth_files = os.listdir(args.ground_truth)
original_image_files = os.listdir(args.images)

ground_truth_files = sorted(ground_truth_files)
original_image_files = sorted(original_image_files)



for threshold in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    predicted_labels = []
    avg_fp = 0
    avg_fn = 0

    avg_npv = 0
    avg_ppv = 0

    avg_dice = 0

    avg_auc = 0

    avg_spec = 0
    avg_sens = 0

    for i in range(len(ground_truth_files)):
        ground_truth_name = args.ground_truth+'/' + ground_truth_files[i]
        original_name = args.images +'/' + original_image_files[i]

        img_patches, img_labels = dataset.image_to_sequential_patches(original_name, ground_truth_name)

        all_predictions = []
        batch_size = 100
        false_positive_patches = 0
        false_negative_patches = 0

        npv = 0
        ppv = 0
        auc = 0

        sens = 0
        spec = 0

        true_positives = 0
        true_negatives = 0

        dice_score = 0

        pos_predictions = 0
        for batch_idx in range(0,len(img_patches),batch_size):
            batch_img_patches = np.array(img_patches[batch_idx:batch_idx+batch_size])
            batch_img_labels = np.array(img_labels[batch_idx:batch_idx+batch_size])

            predictions = model.predict(np.array(batch_img_patches))

            false_positive_patches += np.sum(batch_img_labels[:,1] - predictions[:,1] < 0)
            false_negative_patches += np.sum(batch_img_labels[:,1] - predictions[:,1] > 0)
            # print(batch_img_labels[:,1],predictions[:,1])
            # true negatives
            true_negatives += np.sum(batch_img_labels[:,0] * predictions[:,0] > 0.5)

            # true positives
            true_positives += np.sum(batch_img_labels[:,1] * predictions[:,1] > 0.5)

            pos_predictions += np.sum(predictions)

            all_predictions += predictions[:,1].tolist()
        auc = skm.roc_auc_score(np.array(img_labels)[:,1],all_predictions)
        # import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()

        if (2*true_positives + false_positive_patches+false_negative_patches) == 0:
            dice_score = 0
        else:
            dice_score = 2*true_positives/(2*true_positives + false_positive_patches + false_negative_patches)

        if true_negatives + false_negative_patches == 0:
            npv = 0
        else:
            npv = (true_negatives)/(true_negatives + false_negative_patches)

        if true_positives + false_positive_patches == 0:
            ppv = 0
        else:
            ppv = (true_positives)/(true_positives + false_positive_patches)

        false_negative_patches = false_negative_patches/len(img_patches)
        false_positive_patches = false_positive_patches/len(img_patches)

        if (true_positives + false_negative_patches) > 0:
            sense = true_positives/(true_positives + false_negative_patches)

        if (true_negatives+false_positive_patches) > 0:
            spec = true_negatives/(true_negatives+false_positive_patches)

        avg_fp = (avg_fp*i + false_positive_patches)/(i+1)
        avg_fn = (avg_fn*i + false_negative_patches)/(i+1)

        avg_ppv = (avg_ppv*i + ppv)/(i+1)
        avg_npv = (avg_npv*i + npv)/(i+1)

        avg_spec = (avg_spec*i + spec)/(i+1)
        avg_sens = (avg_sense*i + sense)/(i+1)

        avg_dice = (avg_dice*i + dice_score)/(i+1)

        avg_auc = (avg_auc*i + auc)/(i+1)

        percent_polyp = pos_predictions/len(predictions)
        if percent_polyp > threshold:
            img_label = 1
        else:
            img_label = 0
        # print(i, img_label, 'F+',false_positive_patches, 'F-',false_negative_patches)
        predicted_labels.append(img_label)


    print('Threshold',threshold)

    print('Full Image False Negative Rate', (len(predicted_labels) - np.sum(predicted_labels))/len(predicted_labels))

    print('Full Dataset F+',avg_fp)
    print('Full Dataset F-',avg_fn)

    print('Full Dataset PPV',avg_ppv)
    print('Full Dataset NPV',avg_npv)

    print('DICE SCORE',avg_dice)
    print('AVG AUC',avg_auc)

    print('AVG SENSE',avg_sens)
    print('AVG SPECE',avg_spec)
    # print('AUC',skm.roc_auc_score(np.ones(len(predicted_labels)),predicted_labels))
