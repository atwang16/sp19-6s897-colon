
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.layers.core import Lambda
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import cv2
from keras import models
from keras.models import load_model
import argparse


def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img_path = path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocessing_func(x)
    return x

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='block5_conv4'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instantiate a new model
        new_model = VGG19(weights='imagenet')
    return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def preprocessing_func(X):
    return (X - np.mean(X)) / np.std(X)

def grad_cam(input_model, image, category_index, layer_name):
    model1 = input_model

    nb_classes = 2 # BINARY CLASSIFICATION
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    x = Lambda(target_layer, output_shape=target_category_loss_output_shape)(model1.layers[-1].output)

    model = models.Model(input=model1.input, output=x)

    loss = K.sum(model.layers[-1].output)
    print(loss)
    print(model.summary())
    conv_output =  model.get_layer(layer_name).output
    print("conv output", conv_output)
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap



if __name__ == '__main__':

    # Loading arguments
    parser = argparse.ArgumentParser(description='Visualizing gradients')

    parser.add_argument('--base_path', type=str, required=True, help='Base path.')
    parser.add_argument('--model_path', type=str, required=True, help='Path of model to load')
    parser.add_argument('--image_path', type=str, required=True, default="Path of image to load")
    parser.add_argument('--model_type', type=str, required=True, default="Type of model to load { vgg19 | vgg16 | resnet50 }")

    args = parser.parse_args()

    args.model_path = '/'.join([args.base_path, args.model_path])
    args.image_path = '/'.join([args.base_path, args.image_path])

    # loading image
    preprocessed_input = load_image(args.image_path)

    # loading model
    model = load_model(args.model_path)

    # making prediction for loaded model
    prediction = model.predict(preprocessed_input) #ON
    if round(prediction[0][1]) == 1:
        predicted_class = 'polyp'
        prob = prediction[0][1]
    else:
        predicted_class = 'normal'
        prob = prediction[0][0]
    print('Predicted class:')
    print(f'{predicted_class} with probability {prob}')

    # generating gradcam image
    predicted_class = np.argmax(prediction) # takes index of predicted class
    cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, "block5_conv4")
    cv2.imwrite("gradcam.jpg", cam)

    # generating guided gradcam image
    register_gradient()
    guided_model = modify_backprop(model, 'GuidedBackProp')
    saliency_fn = compile_saliency_function(guided_model)
    saliency = saliency_fn([preprocessed_input, 0])
    gradcam = saliency[0] * heatmap[..., np.newaxis]
    cv2.imwrite("guided_gradcam.jpg", deprocess_image(gradcam))



