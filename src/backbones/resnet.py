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