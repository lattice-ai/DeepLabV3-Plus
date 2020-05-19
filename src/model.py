import tensorflow as tf
from .backbones import AlignedException
from .blocks import AtrousSpatialPyramidPooling


def DeepLabV3Plus(input_shape=(512, 512, 3), model_name='DeepLabV3Plus'):
    # Encoder
    input_tensor = tf.keras.Input(
        shape=input_shape,
        name=model_name + '_Input'
    )
    backbone = AlignedException(input_tensor)
    y = AtrousSpatialPyramidPooling(backbone)
    # Decoder
    encoder_output_shape = tf.keras.backend.int_shape(y)
    y = tf.keras.layers.UpSampling2D(
        [
            encoder_output_shape[1] * 4,
            encoder_output_shape[2] * 4
        ],
        interpolation='bilinear',
        name=model_name + '_Encoder_Output_UpSampling'
    )(y)
    model = tf.keras.Model(input_tensor, y, name=model_name)
    return model
