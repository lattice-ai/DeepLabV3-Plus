import tensorflow as tf


def ConvolutionBlock(
        input_tensor, filters, kernel_size=(3, 3),
        stride=2, use_bias=False, use_bn=True,
        use_activation=True, block_prefix=None):
    y = tf.keras.layers.Conv2D(
        filters, kernel_size, (stride, stride),
        use_bias=use_bias, name=block_prefix + '_Conv2D'
    )(input_tensor)
    y = tf.keras.layers.BatchNormalization(
        name=block_prefix + '_BatchNormalization'
    )(y) if use_bn else y
    y = tf.keras.layers.Activation(
        'relu', name=block_prefix + '_Activation'
    )(y) if use_activation else y
    return y
