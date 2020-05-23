import tensorflow as tf


BASE_WEIGHTS_PATH = ('https://github.com/keras-team/keras-applications/releases/download/resnet/')


WEIGHTS_HASHES = {
    'resnet50': (
        '2cb95161c43110f7111970584f804107',
        '4d473c1dd8becc155b73f8504c6f6626'
    ),
    'resnet101': (
        'f1aeb4b969a6efcfb50fad2f0c20cfc5',
        '88cf7a10940856eca736dc7b7e228a21'
    )
}


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
    if conv_shortcut is True:
        shortcut = tf.keras.layers.Conv2D(
            4 * filters, 1, strides=stride,
            name=name + '_0_conv'
        )(x)
        shortcut = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn'
        )(shortcut)
    else:
        shortcut = x
    x = tf.keras.layers.Conv2D(
        filters, 1, strides=stride,
        name=name + '_1_conv'
    )(x)
    x = tf.keras.layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5,
        name=name + '_1_bn'
    )(x)
    x = tf.keras.layers.Activation(
        'relu', name=name + '_1_relu'
    )(x)
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, padding='SAME',
        name=name + '_2_conv'
    )(x)
    x = tf.keras.layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5,
        name=name + '_2_bn'
    )(x)
    x = tf.keras.layers.Activation(
        'relu', name=name + '_2_relu'
    )(x)
    x = tf.keras.layers.Conv2D(
        4 * filters, 1, name=name + '_3_conv'
    )(x)
    x = tf.keras.layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5,
        name=name + '_3_bn'
    )(x)
    x = tf.keras.layers.Add(name=name + '_add')([shortcut, x])
    x = tf.keras.layers.Activation('relu', name=name + '_out')(x)
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    x = block1(
        x, filters, stride=stride1,
        name=name + '_block1'
    )
    for i in range(2, blocks + 1):
        x = block1(
            x, filters, conv_shortcut=False,
            name=name + '_block' + str(i)
        )
    return x
