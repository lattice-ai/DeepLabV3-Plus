import tensorflow as tf


def ConvolutionBlock(
        input_tensor, filters, kernel_size=3,
        stride=2, use_bias=False, rate=1, use_bn=True,
        use_activation=True, block_prefix=None):
    y = input_tensor
    if stride == 1:
        y = tf.keras.layers.Conv2D(
            filters, (kernel_size, kernel_size), (stride, stride),
            padding='same', use_bias=use_bias, name=block_prefix + '_Conv2D'
        )(y)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        y = tf.keras.layers.ZeroPadding2D((pad_beg, pad_end))(y)
        y = tf.keras.layers.Conv2D(
            filters, (kernel_size, kernel_size), (stride, stride),
            padding='valid', use_bias=False, dilation_rate=(rate, rate),
            name=block_prefix + '_Conv2D'
        )(y)
    y = tf.keras.layers.BatchNormalization(
        name=block_prefix + '_BatchNormalization'
    )(y) if use_bn else y
    y = tf.keras.layers.Activation(
        tf.nn.relu,
        name=block_prefix + '_Activation'
    )(y) if use_activation else y
    return y


def DepthWiseConvolutionBlock(
        input_tensor, filters, kernel_size=3,
        stride=1, rate=1, block_prefix=None):
    padding = None
    y = input_tensor
    if stride == 1:
        padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        y = tf.keras.layers.ZeroPadding2D((pad_beg, pad_end))(y)
        padding = 'valid'
    y = tf.keras.layers.DepthwiseConv2D(
        (kernel_size, kernel_size), strides=(stride, stride),
        dilation_rate=(rate, rate), padding=padding,
        use_bias=False, name=block_prefix + '_DepthwiseConv2D'
    )(y)
    y = tf.keras.layers.BatchNormalization(
        name=block_prefix + '_BatchNormalization'
    )(y)
    y = tf.keras.layers.Activation(
        tf.nn.relu,
        name=block_prefix + '_Activation'
    )(y)
    y = tf.keras.layers.Conv2D(
        filters, (1, 1), padding='same', use_bias=False,
        name=block_prefix + '_Pointwise_Conv2D'
    )(y)
    y = tf.keras.layers.BatchNormalization(
        name=block_prefix + '_BatchNormalization'
    )(y)
    y = tf.keras.layers.Activation(
        tf.nn.relu,
        name=block_prefix + '_Activation'
    )(y)
    return y
