import tensorflow as tf
from .backbones import BACKBONES
from .blocks import AtrousSpatialPyramidPooling


def DeeplabV3Plus(num_classes, height, width, backbone='resnet50'):
    model_input = tf.keras.Input(shape=(height, width, 3))
    resnet50 = BACKBONES[backbone]['model'](
        weights='imagenet',
        include_top=False,
        input_tensor=model_input
    )
    layer = resnet50.get_layer(BACKBONES[backbone]['feature_1']).output
    layer = AtrousSpatialPyramidPooling(layer)
    input_a = tf.keras.layers.UpSampling2D(
        size=(
            height // 4 // layer.shape[1],
            width // 4 // layer.shape[2]
        ),
        interpolation='bilinear'
    )(layer)

    input_b = resnet50.get_layer(BACKBONES[backbone]['feature_2']).output
    input_b = tf.keras.layers.Conv2D(
        48, kernel_size=(1, 1), padding='same',
        kernel_initializer=tf.keras.initializers.he_normal(),
        use_bias=False
    )(input_b)
    input_b = tf.keras.layers.BatchNormalization()(input_b)
    input_b = tf.keras.layers.ReLU()(input_b)

    layer = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])

    layer = tf.keras.layers.Conv2D(
        256, kernel_size=3, padding='same', activation='relu',
        kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False
    )(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    layer = tf.keras.layers.Conv2D(
        256, kernel_size=3, padding='same', activation='relu',
        kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False
    )(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    layer = tf.keras.layers.UpSampling2D(
        size=(
            height // layer.shape[1],
            width // layer.shape[2]
        ),
        interpolation='bilinear'
    )(layer)
    model_output = tf.keras.layers.Conv2D(
        num_classes, kernel_size=(1, 1),
        padding='same'
    )(layer)
    return tf.keras.Model(inputs=model_input, outputs=model_output)
