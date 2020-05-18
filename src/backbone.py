import tensorflow as tf


def get_resnet50(input_height, input_width):
    return tf.keras.applications.resnet50.ResNet50(
        input_shape=(input_height, input_width, 3),
        weights=None, include_top=False
    )
