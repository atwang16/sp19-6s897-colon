
from keras.models import load_model
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import keras.backend as K
import numpy as np
import cv2
import sys
import argparse

def preprocessing_func(X):
    return (X - np.mean(X)) / np.std(X)

if __name__ == '__main__':

	# Loading arguments
	parser = argparse.ArgumentParser(description='Visualizing gradients')

	parser.add_argument('--base_path', type=str, default="/basic_classification", help='Base path')
	parser.add_argument('--model_path', type=str, required=True, help='Path of model to load')
	parser.add_argument('--image_path', type=str, required=True, help="Path of image to load")
	parser.add_argument('--model_type', type=str, required=True, help="Type of model to load { vgg19 | vgg16 | resnet50 }")

	args = parser.parse_args()

	args.model_path = '/'.join([args.base_path, args.model_path])
	args.image_path = '/'.join([args.base_path, args.image_path])

	print('\n=== Loading Model ===\n')

	# loading model
	model = load_model(args.model_path)

	# loading and standardizing image
	img = image.load_img(args.image_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocessing_func(x)

	print('\n=== Making Predictions and Getting Layer ===\n')

	# making class prediction using loaded model
	prediction = model.predict(x)
	print("Predicted", round(prediction[0][1]))
	class_idx = np.argmax(prediction[0])
	class_output = model.output[:, 1]
	model.summary()
	if args.model_type == 'vgg19':
		last_conv_layer = model.get_layer("block5_conv4")
	elif args.model_type == 'vgg16':
		last_conv_layer = model.get_layer("block5_conv3")
	elif args.model_type == 'resnet50':
		last_conv_layer = model.get_layer("resnet50")

	print('\n=== Calculating Gradients and Heatmap ===\n')

	# calculating grads and generating heatmap
	grads = K.gradients(class_output, last_conv_layer.output)[0]
	pooled_grads = K.mean(grads, axis=(0, 1, 2))
	iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
	pooled_grads_value, conv_layer_output_value = iterate([x])
	for i in range(512):
		conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
	heatmap = np.mean(conv_layer_output_value, axis=-1)
	heatmap = np.maximum(heatmap, 0)
	heatmap /= np.max(heatmap)


	print('\n=== Generating Images ===\n')

	# generating original image and superimposed gradcam image
	img = cv2.imread(args.image_path)
	heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
	heatmap = np.uint8(255 * heatmap)
	heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
	superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
	cv2.imshow("Original", img)
	cv2.imshow("GradCam", superimposed_img)
	cv2.waitKey(0)

