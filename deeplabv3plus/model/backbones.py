import tensorflow as tf


BACKBONES = {
    'resnet50': {
        'model': tf.keras.applications.ResNet50,
        'feature_1': 'conv4_block6_2_relu',
        'feature_2': 'conv2_block3_2_relu'
    },
    'mobilenetv2': {
        'model': tf.keras.applications.MobileNetV2,
        'feature_1': 'out_relu',
        'feature_2': 'block_3_depthwise_relu'
    }
}
