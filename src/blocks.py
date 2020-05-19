import tensorflow as tf


def ASPPConvBlock(
        input_tensor, filters, kernel_size=3,
        padding='same', use_bias=False, block_prefix=None):
    y = tf.keras.layers.Conv2D(
        filters, kernel_size=(kernel_size, kernel_size), padding=padding,
        use_bias=use_bias, name=block_prefix + '_Conv2D'
    )(input_tensor)
    y = tf.keras.layers.BatchNormalization(
        name=block_prefix + '_BatchNormalization'
    )(y)
    y = tf.keras.layers.Activation(
        tf.nn.relu, name=block_prefix + '_Activation'
    )(y)
    return y
