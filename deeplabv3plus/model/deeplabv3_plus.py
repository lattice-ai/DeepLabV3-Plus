"""Module providing the DeeplabV3+ network architecture as a tf.keras.Model.
"""

import tensorflow as tf
from typing import List

from .backbones import BACKBONES
from .blocks import ConvolutionBlock, aspp_block


def DeeplabV3Plus(input_shape: List[int], backbone: str, n_classes: int):
    model_input = tf.keras.Input(shape=input_shape)
    backbone_model = BACKBONES[backbone]['model'](
        weights='imagenet', include_top=False, input_tensor=model_input
    )
    print(BACKBONES[backbone]['feature_1'])
    layer = backbone_model.get_layer(BACKBONES[backbone]['feature_1']).output
    layer = aspp_block(layer)
    input_a = tf.keras.layers.UpSampling2D(
        size=(
            input_shape[0] // 4 // layer.shape[1],
            input_shape[1] // 4 // layer.shape[2]
        ), interpolation='bilinear'
    )(layer)

    input_b = backbone_model.get_layer(BACKBONES[backbone]['feature_2']).output
    input_b = ConvolutionBlock(
        n_filters=48, kernel_size=(1, 1), padding='same',
        kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False
    )(input_b)

    layer = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])

    layer = ConvolutionBlock(
        n_filters=256, kernel_size=3,
        padding='same', use_bias=False,
        kernel_initializer=tf.keras.initializers.he_normal(),
    )(layer)
    layer = ConvolutionBlock(
        n_filters=256, kernel_size=3,
        padding='same', use_bias=False,
        kernel_initializer=tf.keras.initializers.he_normal(),
    )(layer)
    layer = tf.keras.layers.UpSampling2D(
        size=(
            input_shape[0] // layer.shape[1],
            input_shape[1] // layer.shape[2]
        ), interpolation='bilinear'
    )(layer)
    model_output = tf.keras.layers.Conv2D(
        n_classes, kernel_size=(1, 1),
        padding='same'
    )(layer)
    return tf.keras.Model(
        inputs=model_input, outputs=model_output
    )
