import tensorflow as tf
from tensorflow.keras import backend as K


def AtrousSpatialPyramidPooling(input_tensor):
    dims = K.int_shape(input_tensor)

    x = tf.keras.layers.AveragePooling2D(
        pool_size=(dims[-3], dims[-2])
    )(input_tensor)

    x = tf.keras.layers.Conv2D(
        256, kernel_size=1, padding='same',
        kernel_initializer=tf.keras.initializers.he_normal()
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    pool = tf.keras.layers.UpSampling2D(
        size=(
            dims[-3] // x.shape[1],
            dims[-2] // x.shape[2]
        ),
        interpolation='bilinear'
    )(x)

    x = tf.keras.layers.Conv2D(
        256, kernel_size=1,
        dilation_rate=1, padding='same',
        kernel_initializer=tf.keras.initializers.he_normal(),
        use_bias=False
    )(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    out_1 = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        256, kernel_size=3,
        dilation_rate=6, padding='same',
        kernel_initializer=tf.keras.initializers.he_normal(),
        use_bias=False
    )(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    out_6 = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        256, kernel_size=3,
        dilation_rate=12, padding='same',
        kernel_initializer=tf.keras.initializers.he_normal(),
        use_bias=False
    )(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    out_12 = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        256, kernel_size=3,
        dilation_rate=18, padding='same',
        kernel_initializer=tf.keras.initializers.he_normal(),
        use_bias=False
    )(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    out_18 = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Concatenate(axis=-1)([
        pool, out_1, out_6, out_12, out_18
    ])

    x = tf.keras.layers.Conv2D(
        256, kernel_size=1,
        dilation_rate=1, padding='same',
        kernel_initializer=tf.keras.initializers.he_normal(),
        use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    return tf.keras.layers.ReLU()(x)
