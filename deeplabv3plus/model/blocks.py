import tensorflow as tf
from tensorflow.keras import backend as K


 


def AtrousSpatialPyramidPooling(input_tensor):
    dims = K.int_shape(input_tensor)

    x = tf.keras.layers.AveragePooling2D(
        pool_size=(dims[-3], dims[-2])
    )(input_tensor)

    x = ConvBlock(
        x, 256, kernel_size=1, padding='same',dilation_rate=1,
        kernel_initializer=tf.keras.initializers.he_normal()
    )

    pool = tf.keras.layers.UpSampling2D(
        size=(
            dims[-3] // x.shape[1],
            dims[-2] // x.shape[2]
        ),
        interpolation='bilinear'
    )(x)

    out_1 = ConvBlock(
        input_tensor, 256, kernel_size=1,
        dilation_rate=1, padding='same', use_bias=False,
        kernel_initializer=tf.keras.initializers.he_normal()
    )

    out_6 = ConvBlock(
        input_tensor, 256, kernel_size=3,
        dilation_rate=6, padding='same', use_bias=False,
        kernel_initializer=tf.keras.initializers.he_normal()
    )

    out_12 = ConvBlock(
        input_tensor, 256, kernel_size=3,
        dilation_rate=12, padding='same', use_bias=False,
        kernel_initializer=tf.keras.initializers.he_normal()
    )

    out_18 = ConvBlock(
        input_tensor, 256, kernel_size=3,
        dilation_rate=18, padding='same', use_bias=False,
        kernel_initializer=tf.keras.initializers.he_normal()
    )

    x = tf.keras.layers.Concatenate(axis=-1)([
        pool, out_1, out_6, out_12, out_18
    ])

    x = ConvBlock(
        x, 256, kernel_size=1,
        dilation_rate=1, padding='same', use_bias=False,
        kernel_initializer=tf.keras.initializers.he_normal()
    )
    return x