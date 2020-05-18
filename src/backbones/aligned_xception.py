import tensorflow as tf


def ConvolutionBlock(
        input_tensor, filters, kernel_size=3,
        stride=2, use_bias=False, rate=1, use_bn=True,
        use_activation=True, block_prefix=None):
    y = input_tensor
    if stride == 1:
        y = tf.keras.layers.Conv2D(
            filters, (kernel_size, kernel_size), (stride, stride),
            use_bias=use_bias, name=block_prefix + '_Conv2D'
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
