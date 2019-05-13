import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import MaxPool2D, Input, Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
#
# model2 = VGG19(include_top = False)
# model2.summary()
# import pdb; pdb.set_trace()
def patch_vgg(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv2D(8,
           kernel_size=8,
           strides=1,
           padding="same",
           kernel_initializer="he_normal")(inputs)
    x = Conv2D(8,#extra
           kernel_size=8,
           strides=1,
           padding="same",
           kernel_initializer="he_normal")(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Conv2D(16,
           kernel_size=4,
           strides=1,
           padding="same",
           kernel_initializer="he_normal")(x)
    x = Conv2D(16,#extra
           kernel_size=4,
           strides=1,
           padding="same",
           kernel_initializer="he_normal")(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(1560, activation="relu")(x)# was 500
    x = Dense(1560, activation="relu")(x)# was 500
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def patch_vgg_regress(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # model = VGG19(include_top = False,weights='imagenet', input_shape = input_shape)
    # for idx in range(len(model.layers)):
    #     model.layers[idx].trainable = False
    #
    # x = model.layers[1](inputs)
    #
    # for idx in range(2,4):
    #     x = model.layers[idx](x)

    x = Conv2D(64,
           kernel_size=3,
           strides=1,
           padding="same",
           kernel_initializer="he_normal")(inputs)
    x = Conv2D(64,
           kernel_size=3,
           strides=1,
           padding="same",
           kernel_initializer="he_normal")(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(1560, activation="relu")(x)
    x = Dense(1560, activation="relu")(x)
    outputs = Dense(1, activation="relu")(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def patch_vgg_pretrained(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    model = VGG19(include_top = False,weights='imagenet', input_shape = input_shape)
    for idx in range(len(model.layers)):
        model.layers[idx].trainable = False
    #
    # x = model.layers[1](inputs)
    #
    # for idx in range(2,4):
    #     x = model.layers[idx](x)

    x = model(inputs)
    x = Flatten(name='flatten')(x)
    x = Dense(1560, activation="relu")(x)
    x = Dense(1560, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model
