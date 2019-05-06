#!/usr/bin/env python3

import sys, os
from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
from keras.preprocessing import image


def load_images_from_directory(directory, n_samples):

	image_arrays = []
	files = [f for f in listdir(directory) if isfile(join(directory, f))]
	if '.DS_Store' in files:
		files.remove('.DS_Store')

	i = 0
	while True:
		img_path = files[i]
		path = os.path.join(directory, img_path)
		img = image.load_img(path, target_size=(224, 224)) # Resizing the images to (224, 224)
		x = image.img_to_array(img)
		yield x
		i += 1


def pixel_standardize(directory, n_samples):

	image_generator = load_images_from_directory(directory, n_samples)
	image_arrays=[]
	for i in range(n_samples):
		image = next(image_generator)
		[r, g, b] = np.dsplit(np.array(image), np.array(image).shape[-1])
		r = np.subtract(r, np.mean(r)) / np.std(r)
		g = np.subtract(g, np.mean(g)) / np.std(g)
		b = np.subtract(b, np.mean(b)) / np.std(b)
		image = np.dstack((r, g, b))
		image_arrays.append(image)

	return np.array(image_arrays)

trainX = pixel_standardize('/Users/exiao/MIT/hst.956_final_project/CVC-ClinicDB_converted/train/polyp', 612)
