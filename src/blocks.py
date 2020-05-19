import tensorflow as tf


def ASPPConvBlock(
        input_tensor, filters, kernel_size=3, rate=1,
        padding='same', use_bias=False, block_prefix=None):
    y = tf.keras.layers.Conv2D(
        filters, kernel_size=(kernel_size, kernel_size), padding=padding,
        dilation_rate=rate, use_bias=use_bias, name=block_prefix + '_Conv2D'
    )(input_tensor)
    y = tf.keras.layers.BatchNormalization(
        name=block_prefix + '_BatchNormalization'
    )(y)
    y = tf.keras.layers.Activation(
        tf.nn.relu, name=block_prefix + '_Activation'
    )(y)
    return y


def AtrousSpatialPyramidPooling(input_tensor, block_prefix='ASPP'):
    input_shape = tf.keras.backend.int_shape(input_tensor)
    # 1x1 Conv
    y_1x1 = ASPPConvBlock(
        input_tensor, 256, kernel_size=1,
        block_prefix=block_prefix + '_1x1_Conv_Block'
    )
    # 3x3 Conv, rate 6
    y_3x3_r6 = ASPPConvBlock(
        input_tensor, 256, kernel_size=3, rate=6,
        block_prefix=block_prefix + '_3x3_Conv_Block_Rate_6'
    )
    # 3x3 Conv, rate 12
    y_3x3_r12 = ASPPConvBlock(
        input_tensor, 256, kernel_size=3, rate=12,
        block_prefix=block_prefix + '_3x3_Conv_Block_Rate_12'
    )
    # 3x3 Conv, rate 18
    y_3x3_r18 = ASPPConvBlock(
        input_tensor, 256, kernel_size=3, rate=18,
        block_prefix=block_prefix + '_3x3_Conv_Block_Rate_18'
    )
    # Image Pooling
    y_image_pooling = tf.keras.layers.AveragePooling2D(
        pool_size=(
            input_shape[1],
            input_shape[2]
        ),
        name=block_prefix + '_Average_Pooling'
    )(input_tensor)
    y_image_pooling = ASPPConvBlock(
        y_image_pooling, 256, kernel_size=1,
        block_prefix=block_prefix + '_Image_Pooling_Conv_Block'
    )
    y_image_pooling = tf.keras.layers.UpSampling2D(
        size=[input_shape[1], input_shape[2]],
        interpolation='bilinear'
    )(y_image_pooling)
    # Concatenation
    y = tf.keras.layers.Concatenate(
        name=block_prefix + '_Concatenate'
    )([
        y_1x1, y_3x3_r6, y_3x3_r12,
        y_3x3_r18, y_image_pooling
    ])
    y = ASPPConvBlock(
        y, 256, kernel_size=1, rate=1,
        block_prefix=block_prefix + '_Final_ASPP_Conv_Block'
    )
    return y
