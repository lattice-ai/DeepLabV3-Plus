import tensorflow as tf
from .backbones import AlignedException
from .blocks import AtrousSpatialPyramidPooling


def DeepLabV3Plus(input_shape=(512, 512, 3), model_name='DeepLabV3Plus'):
    input_tensor = tf.keras.Input(
        shape=input_shape,
        name=model_name + '_Input'
    )
    backbone = AlignedException(input_tensor)
    y = AtrousSpatialPyramidPooling(backbone)
    model = tf.keras.Model(input_tensor, y, name=model_name)
    return model
