import tensorflow as tf
from .aligned_xception import (
    ConvolutionBlock,
    DepthWiseConvolutionBlock,
    EntryFlowBlock,
    MiddleFlowBlock
)


def AlignedException(input_tensor, block_prefix='Aligned_Exception'):
    ## Entry Flow
    y = ConvolutionBlock(input_tensor, 32, block_prefix=block_prefix + '_Entry_Flow_Conv_1')
    y = ConvolutionBlock(y, 64, stride=1, block_prefix=block_prefix + '_Entry_Flow_Conv_2')
    y = EntryFlowBlock(y, 128, block_prefix=block_prefix + '_Entry_Flow_Block_1')
    y = EntryFlowBlock(y, 256, block_prefix=block_prefix + '_Entry_Flow_Block_2')
    y = EntryFlowBlock(y, 728, block_prefix=block_prefix + '_Entry_Flow_Block_3')
    ## Middle Flow
    for i in range(1, 17):
        y = MiddleFlowBlock(y, block_prefix=block_prefix + '_Middle_Flow_' + str(i))
    ## Exit Flow
    exit_flow_input = y
    y = DepthWiseConvolutionBlock(
        y, 728, stride=1,
        block_prefix=block_prefix + 'Exit_Flow_DepthWiseConvBlock_1'
    )
    y = DepthWiseConvolutionBlock(
        y, 1024, stride=1,
        block_prefix=block_prefix + 'Exit_Flow_DepthWiseConvBlock_2'
    )
    y = DepthWiseConvolutionBlock(
        y, 1024, stride=2,
        block_prefix=block_prefix + 'Exit_Flow_DepthWiseConvBlock_3'
    )
    exit_flow_residual = ConvolutionBlock(
        exit_flow_input, 1024, kernel_size=1, stride=2,
        block_prefix=block_prefix + '_Exit_Flow_Residual'
    )
    y = tf.keras.layers.Add(name=block_prefix + '_Exit_Flow_Add')([y, exit_flow_residual])
    y = DepthWiseConvolutionBlock(
        y, 1536, stride=1,
        block_prefix=block_prefix + 'Exit_Flow_DepthWiseConvBlock_4'
    )
    y = DepthWiseConvolutionBlock(
        y, 1536, stride=2,
        block_prefix=block_prefix + 'Exit_Flow_DepthWiseConvBlock_5'
    )
    y = DepthWiseConvolutionBlock(
        y, 2048, stride=1,
        block_prefix=block_prefix + 'Exit_Flow_DepthWiseConvBlock_6'
    )
    return y


def get_resnet_101(input_shape, weights='imagenet'):
    return tf.keras.applications.resnet.ResNet101(
        input_shape=input_shape, weights=weights, include_top=False
    )
