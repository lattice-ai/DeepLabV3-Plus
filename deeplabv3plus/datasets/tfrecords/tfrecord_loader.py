import tensorflow as tf
from typing import List


class TFRecordLoader:

    def __init__(self, image_size):
        self.image_size = image_size

    def parse_image(self, image):
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = (tf.image.resize(
            images=image, size=[self.image_size, self.image_size]
        ))
        image = tf.cast(image, tf.float32) / 127.5 - 1
        return image

    def parse_label(self, label):
        label = tf.image.decode_png(label, channels=1)
        label.set_shape([None, None, 1])
        label = (tf.image.resize(
            images=label, size=[self.image_size, self.image_size], method="nearest"
        ))
        label = tf.cast(label, tf.float32)
        return label

    @tf.function
    def map_function(self, example):
        features_desc = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string)
        }
        features = tf.io.parse_single_example(example, features_desc)
        image = features.pop('image')
        image = self.parse_image(image)
        label = features.pop('label')
        label = self.parse_image(label)
        return image, label

    def get_dataset(self, train_tfrecord_files: List[str], ignore_order: bool = False):
        options = tf.data.Options()
        options.experimental_deterministic = False
        dataset = tf.data.TFRecordDataset(
            train_tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)
        dataset = dataset.with_options(options) if ignore_order else dataset
        dataset = dataset.map(
            map_func=self.map_function, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset
