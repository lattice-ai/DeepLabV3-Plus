import tensorflow as tf
from .backbones import AlignedException, get_resnet_101
from .blocks import AtrousSpatialPyramidPooling, BilinearUpsample


def DeepLabV3Plus(input_shape=(512, 512, 3), backbone='resnet101', model_name='DeepLabV3Plus'):
    # Encoder
    if backbone == 'resnet101':
        backbone = get_resnet_101(input_shape=input_shape, weights=None)
        input_tensor = backbone.input
        y = AtrousSpatialPyramidPooling(backbone.get_layer('conv5_block3_out').output)
        decoder_input_tensor = backbone.get_layer('conv2_block3_add').output
    elif backbone == 'alignedexception' or backbone == 'exception':
        input_tensor = tf.keras.Input(
            shape=input_shape,
            name=model_name + '_Input'
        )
        backbone = AlignedException(input_tensor)
        y = AtrousSpatialPyramidPooling(backbone)
    encoder_y = BilinearUpsample(
        tensor=y, size=[
            input_shape[0] // 4,
            input_shape[1] // 4
        ], name='Encoder_Output_Upsample'
    )
    # Decoder
    decoder_y = tf.keras.layers.Conv2D(
        filters=48, kernel_size=1, padding='same',
        name='Decoder_Conv_1x1_1', use_bias=False
    )(decoder_input_tensor)
    decoder_y = tf.keras.layers.BatchNormalization(
        name='Decoder_BacthNorm_1'
    )(decoder_y)
    decoder_y = tf.keras.layers.Activation(
        tf.nn.relu, name='Decoder_Activation_1'
    )(decoder_y)
    print(tf.keras.backend.int_shape(encoder_y), tf.keras.backend.int_shape(decoder_y))
    y = tf.keras.layers.Concatenate(
        name='Encoder_Decoder_Concatenation'
    )([encoder_y, decoder_y])
    model = tf.keras.Model(input_tensor, y, name=model_name)
    return model
