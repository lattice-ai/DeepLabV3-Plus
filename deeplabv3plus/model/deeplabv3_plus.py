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
    def __init__(self, num_classes, height, width, backbone='resnet50'):
        super(DeeplabV3Plus, self).__init__()

        self.num_classes = num_classes
        self.height, self.width = height, width
        self.backbone = backbone
        self.aspp = None
        self.input_b_conv, self.conv1, self.conv2 = None, None, None
        self.out_conv = None

        self._built = False

    def _build(self):
        if self._built:
            return

        self._built = True

        self.aspp = AtrousSpatialPyramidPooling()

        self.input_b_conv = ConvBlock(
            48, kernel_size=(1, 1), padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            use_bias=False, dilation_rate=1)

        self.conv1 = ConvBlock(
            256, kernel_size=3, padding='same', conv_activation='relu',
            kernel_initializer=tf.keras.initializers.he_normal(),
            use_bias=False, dilation_rate=1)

        self.conv2 = ConvBlock(
            256, kernel_size=3, padding='same', conv_activation='relu',
            kernel_initializer=tf.keras.initializers.he_normal(),
            use_bias=False, dilation_rate=1)

        self.out_conv = tf.keras.layers.Conv2D(self.num_classes,
                                               kernel_size=(1, 1),
                                               padding='same')

    def call(self, inputs, training=None, mask=None):
        self._build()

        inputs = tf.keras.Input(shape=(self.height, self.width, 3),
                                tensor=inputs)

        backbone_model = BACKBONES[self.backbone]['model'](
            weights='imagenet', input_tensor=inputs)

        tensor = backbone_model.get_layer(
            BACKBONES[self.backbone]['feature_1']).output
        tensor = self.aspp(tensor)

        input_a = tf.keras.layers.UpSampling2D(
            size=(
                self.height
                // 4 // tensor.shape[1],
                self.width
                // 4 // tensor.shape[2]
            ),
            interpolation='bilinear'
        )(tensor)

        input_b = backbone_model.get_layer(
            BACKBONES[self.backbone]['feature_2']).output
        input_b = self.input_b_conv(input_b)

        tensor = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
        tensor = self.conv2(self.conv1(tensor))

        tensor = tf.keras.layers.UpSampling2D(
            size=(
                self.height
                // tensor.shape[1],
                self.width
                // tensor.shape[2]
            ),
            interpolation='bilinear'
        )
        return self.out_conv(tensor)
