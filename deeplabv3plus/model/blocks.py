"""Module providing building blocks for the DeepLabV3+ netowork architecture.
"""

import tensorflow as tf


class ConvolutionBlock(tf.keras.layers.Layer):

    def __init__(
            self, n_filters: int, kernel_size: int, padding, kernel_initializer,
            dilation_rate=1, use_bias: bool = True, apply_batch_norm: bool = True,
            apply_activation: bool = True):
        super(ConvolutionBlock, self).__init__()
        self.apply_batch_norm = apply_batch_norm
        self.apply_activation = apply_activation
        self.convolution = tf.keras.layers.Conv2D(
            filters=n_filters, kernel_size=kernel_size, padding=padding,
            kernel_initializer=kernel_initializer, use_bias=use_bias, dilation_rate=dilation_rate
        )
        self.batch_normalization = tf.keras.layers.BatchNormalization() if self.apply_batch_norm else None

    def call(self, inputs, **kwargs):
        x = self.convolution(inputs)
        x = self.batch_normalization(x) if self.apply_batch_norm else x
        x = tf.nn.relu(x) if self.apply_activation else x
        return x


def aspp_block(input_tensor):
    dims = tf.keras.backend.int_shape(input_tensor)

    layer = tf.keras.layers.AveragePooling2D(
        pool_size=(dims[-3], dims[-2])
    )(input_tensor)
    layer = ConvolutionBlock(
        n_filters=256, kernel_size=1,
        padding='same', kernel_initializer='he_normal'
    )(layer)
    out_pool = tf.keras.layers.UpSampling2D(
        size=(
            dims[-3] // layer.shape[1],
            dims[-2] // layer.shape[2]
        ), interpolation='bilinear'
    )(layer)

    layer = ConvolutionBlock(
        n_filters=256, kernel_size=1,
        dilation_rate=1, padding='same',
        kernel_initializer='he_normal', use_bias=False,
        apply_batch_norm=True, apply_activation=False
    )(input_tensor)
    out_1 = tf.nn.relu(layer)

    layer = ConvolutionBlock(
        n_filters=256, kernel_size=3,
        dilation_rate=6, padding='same',
        kernel_initializer='he_normal', use_bias=False,
        apply_batch_norm=True, apply_activation=False
    )(input_tensor)
    out_6 = tf.keras.layers.ReLU()(layer)

    layer = ConvolutionBlock(
        n_filters=256, kernel_size=3,
        dilation_rate=12, padding='same',
        kernel_initializer='he_normal', use_bias=False,
        apply_batch_norm=True, apply_activation=False
    )(input_tensor)
    out_12 = tf.keras.layers.ReLU()(layer)

    layer = ConvolutionBlock(
        n_filters=256, kernel_size=3,
        dilation_rate=18, padding='same',
        kernel_initializer='he_normal', use_bias=False,
        apply_batch_norm=True, apply_activation=False
    )(input_tensor)
    out_18 = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Concatenate(axis=-1)([
        out_pool, out_1, out_6, out_12, out_18
    ])

    model_output = ConvolutionBlock(
        n_filters=256, kernel_size=1,
        dilation_rate=1, padding='same',
        kernel_initializer='he_normal', use_bias=False
    )(layer)

    return model_output
