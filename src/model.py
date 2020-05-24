import tensorflow as tf
from .backbones import AlignedException, get_resnet_101
from .blocks import AtrousSpatialPyramidPooling, BilinearUpsample


def DeepLabV3Plus(input_shape=(512, 512, 3), backbone='resnet101', model_name='DeepLabV3Plus'):
    # Encoder
    if backbone == 'resnet101':
        backbone = get_resnet_101(input_shape=input_shape, weights=None)
        input_tensor = backbone.input
        y = AtrousSpatialPyramidPooling(backbone.get_layer('conv5_block3_out').output)
    elif backbone == 'alignedexception' or backbone == 'exception':
        input_tensor = tf.keras.Input(
            shape=input_shape,
            name=model_name + '_Input'
        )
        backbone = AlignedException(input_tensor)
        y = AtrousSpatialPyramidPooling(backbone)
    y = BilinearUpsample(
        tensor=y, size=[
            input_shape[0] // 4,
            input_shape[1] // 4
        ], name='Encoder_Output_Upsample'
    )
    model = tf.keras.Model(input_tensor, y, name=model_name)
    return model
