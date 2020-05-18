from .aligned_xception import (
    ConvolutionBlock,
    DepthWiseConvolutionBlock,
    EntryFlowBlock
)


def AlignedException(input_tensor, block_prefix='Aligned_Exception'):
    y = ConvolutionBlock(input_tensor, 32, block_prefix=block_prefix + 'Entry_Flow_Conv_1')
    y = ConvolutionBlock(y, 64, stride=1, block_prefix=block_prefix + 'Entry_Flow_Conv_2')
    y = EntryFlowBlock(y, 128, block_prefix=block_prefix + 'Entry_Flow_Block_1')
    y = EntryFlowBlock(y, 256, block_prefix=block_prefix + 'Entry_Flow_Block_2')
    y = EntryFlowBlock(y, 728, block_prefix=block_prefix + 'Entry_Flow_Block_3')
    return y
