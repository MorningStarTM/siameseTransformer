import tensorflow as tf
from keras import layers
from keras.models import Model
import numpy as np
from .vtransformer import CNN_ViT


hp = {}
hp['image_size'] = 256
hp['num_channels'] = 3
hp['patch_size'] = 32
hp['num_patches'] = (hp['image_size']**2) // (hp["patch_size"]**2)
hp["flat_patches_shape"] = (hp["num_patches"], hp['patch_size']*hp['patch_size']*hp["num_channels"])
hp['batch_size'] = 2
hp['lr'] = 1e-5
hp["num_epochs"] = 30
hp['num_classes'] = 3
hp["num_layers"] = 3
hp["hidden_dim"] = 256
hp["mlp_dim"] = 256
hp['num_heads'] = 3
hp['dropout_rate'] = 0.1
hp['class_names'] = ["s1", "s2", "s3"]

vit = CNN_ViT(hp)


def euclidean_distance(vects):
  x,y = vects
  sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
  return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))


def SiameseTransformer(target_size):
    input_1 = layers.Input(shape=target_size)
    input_2 = layers.Input(shape=target_size)

    tower_1 = vit(input_1)
    tower_2 = vit(input_2)

    merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
    normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
    output_layer = layers.Dense(1, activation='sigmoid')(normal_layer)

    siamese = Model([input_1, input_2], output_layer)

    return siamese


