"""Module providing the DeeplabV3+ network architecture as a tf.keras.Model.
"""

import tensorflow as tf

from .backbones import BACKBONES
from .blocks import (AtrousSpatialPyramidPooling,
                     ConvBlock)


# !pylint:disable=too-many-ancestors, too-many-instance-attributes
class DeeplabV3Plus(tf.keras.Model):
    """DeeplabV3+ network architecture provider tf.keras.Model implementation.
    Args:
        num_classes:
            number of segmentation classes, effectively - number of output
            filters
        height, width:
            expected height, width of image
        backbone:
            backbone to be used
    """
    def __init__(self, num_classes, backbone='resnet50', **kwargs):
        super(DeeplabV3Plus, self).__init__()

        self.num_classes = num_classes
        self.backbone = backbone
        self.aspp = None
        self.backbone_feature_1, self.backbone_feature_2 = None, None
        self.input_a_upsampler_getter = None
        self.otensor_upsampler_getter = None
        self.input_b_conv, self.conv1, self.conv2, self.out_conv = (None,
                                                                    None,
                                                                    None,
                                                                    None)

    @staticmethod
    def _get_conv_block(filters, kernel_size, conv_activation=None):
        return ConvBlock(filters, kernel_size=kernel_size, padding='same',
                         conv_activation=conv_activation,
                         kernel_initializer=tf.keras.initializers.he_normal(),
                         use_bias=False, dilation_rate=1)

    @staticmethod
    def _get_upsample_layer_fn(input_shape, factor: int):
        return lambda fan_in_shape: \
            tf.keras.layers.UpSampling2D(
                size=(
                    input_shape[1]
                    // factor // fan_in_shape[1],
                    input_shape[2]
                    // factor // fan_in_shape[2]
                ),
                interpolation='bilinear'
            )

    def _get_backbone_feature(self, feature: str,
                              input_shape) -> tf.keras.Model:
        input_layer = tf.keras.Input(shape=input_shape[1:])

        backbone_model = BACKBONES[self.backbone]['model'](
            input_tensor=input_layer, weights='imagenet', include_top=False)

        output_layer = backbone_model.get_layer(
            BACKBONES[self.backbone][feature]).output
        return tf.keras.Model(inputs=input_layer, outputs=output_layer)

    def build(self, input_shape):
        self.backbone_feature_1 = self._get_backbone_feature('feature_1',
                                                             input_shape)
        self.backbone_feature_2 = self._get_backbone_feature('feature_2',
                                                             input_shape)

        self.input_a_upsampler_getter = self._get_upsample_layer_fn(
            input_shape, factor=4)

        self.aspp = AtrousSpatialPyramidPooling()

        self.input_b_conv = DeeplabV3Plus._get_conv_block(48,
                                                          kernel_size=(1, 1))

        self.conv1 = DeeplabV3Plus._get_conv_block(256, kernel_size=3,
                                                   conv_activation='relu')

        self.conv2 = DeeplabV3Plus._get_conv_block(256, kernel_size=3,
                                                   conv_activation='relu')

        self.otensor_upsampler_getter = self._get_upsample_layer_fn(
            input_shape, factor=1)

        self.out_conv = tf.keras.layers.Conv2D(self.num_classes,
                                               kernel_size=(1, 1),
                                               padding='same')

    def call(self, inputs, training=None, mask=None):
        input_a = self.backbone_feature_1(inputs)

        input_a = self.aspp(input_a)
        input_a = self.input_a_upsampler_getter(input_a.shape)(input_a)

        input_b = self.backbone_feature_2(inputs)
        input_b = self.input_b_conv(input_b)

        tensor = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
        tensor = self.conv2(self.conv1(tensor))

        tensor = self.otensor_upsampler_getter(tensor.shape)(tensor)
        return self.out_conv(tensor)
