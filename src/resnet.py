import tensorflow as tf
from keras import layers
from keras.models import Model
import numpy as np



def resnet_block(x, filters, strides=1):
    identity = x

    x = layers.Conv2D(filters, kernel_size=5, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, kernel_size=5, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if strides > 1:
        identity = layers.Conv2D(filters, kernel_size=1, strides=strides, padding='same')(identity)
        identity = layers.BatchNormalization()(identity)

    x = layers.Add()([x, identity])
    x = layers.Activation('relu')(x)
    return x


def build_resnet(input_shape):

    x = layers.Conv2D(32, kernel_size=7, strides=2, padding='same')(input_shape)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    
    x = resnet_block(x, filters=32)
    x = resnet_block(x, filters=32)

    x = resnet_block(x, filters=64, strides=2)
    x = resnet_block(x, filters=64)

    x = resnet_block(x, filters=128, strides=2)
    x = resnet_block(x, filters=128)
    
    x = resnet_block(x, filters=256, strides=2)
    x = resnet_block(x, filters=256)
    
    x = resnet_block(x, filters=256, strides=2)
    x = resnet_block(x, filters=256)
    
    return x