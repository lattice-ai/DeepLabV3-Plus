import tensorflow as tf
from tensorflow.keras import backend as K


def AtrousSpatialPyramidPooling(input_tensor):
    dims = K.int_shape(input_tensor)

    layer = tf.keras.layers.AveragePooling2D(
        pool_size=(dims[-3], dims[-2])
    )(input_tensor)

    layer = tf.keras.layers.Conv2D(
        256, kernel_size=1, padding='same',
        kernel_initializer=tf.keras.initializers.he_normal()
    )(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)

    out_pool = tf.keras.layers.UpSampling2D(
        size=(
            dims[-3] // layer.shape[1],
            dims[-2] // layer.shape[2]
        ),
        interpolation='bilinear'
    )(layer)

    layer = tf.keras.layers.Conv2D(
        256, kernel_size=1,
        dilation_rate=1, padding='same',
        kernel_initializer=tf.keras.initializers.he_normal(),
        use_bias=False
    )(input_tensor)
    layer = tf.keras.layers.BatchNormalization()(layer)
    out_1 = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(
        256, kernel_size=3,
        dilation_rate=6, padding='same',
        kernel_initializer=tf.keras.initializers.he_normal(),
        use_bias=False
    )(input_tensor)
    layer = tf.keras.layers.BatchNormalization()(layer)
    out_6 = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(
        256, kernel_size=3,
        dilation_rate=12, padding='same',
        kernel_initializer=tf.keras.initializers.he_normal(),
        use_bias=False
    )(input_tensor)
    layer = tf.keras.layers.BatchNormalization()(layer)
    out_12 = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(
        256, kernel_size=3,
        dilation_rate=18, padding='same',
        kernel_initializer=tf.keras.initializers.he_normal(),
        use_bias=False
    )(input_tensor)
    layer = tf.keras.layers.BatchNormalization()(layer)
    out_18 = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Concatenate(axis=-1)([
        out_pool, out_1, out_6, out_12, out_18
    ])

    layer = tf.keras.layers.Conv2D(
        256, kernel_size=1,
        dilation_rate=1, padding='same',
        kernel_initializer=tf.keras.initializers.he_normal(),
        use_bias=False
    )(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)

    return tf.keras.layers.ReLU()(layer)
