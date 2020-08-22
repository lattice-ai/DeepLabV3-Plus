"""Module providing a data loader for DeepLabV3+."""

import tensorflow as tf

from deeplabv3plus.datasets.augmentations import Augmentation


class GenericDataLoader:
    """Class for loading data from the specified image file globs, and
    converting them to a tf.data pipeline."""
    def __init__(self, config):
        self.config = config

        # primarily a wrapper for a functional API
        # low memory footprint, hence safe to load eagerly
        self.augmentation = Augmentation(config)

        self.assert_dataset_config()

    def assert_dataset_config(self):
        """Asserts dataset config."""
        assert 'images' in self.config and 'labels' in self.config
        assert len(self.config['images']) == len(self.config['labels'])
        print('[+] Train Images are good to go!')

    def __len__(self):
        return len(self.config['images'])

    def read_img(self, image_path, mask=False):
        """Reads image from the given path.

        Args:
            image_path:
                path to read image file from
            mask:
                boolean stating whether we are reading a normal RGB image or a
                binary segmentation mask.
        """
        image = tf.io.read_file(image_path)
        if mask:
            image = tf.image.decode_png(image, channels=1)
            image.set_shape([None, None, 1])
            image = (tf.image.resize(
                images=image, size=[
                    self.config['height'],
                    self.config['width']
                ], method="nearest"
            ))
            image = tf.cast(image, tf.float32)
        else:
            image = tf.image.decode_png(image, channels=3)
            image.set_shape([None, None, 3])
            image = (tf.image.resize(
                images=image, size=[
                    self.config['height'],
                    self.config['width']
                ]
            ))
            image = tf.cast(image, tf.float32) / 127.5 - 1
        return image

    def _map_function(self, image_list, mask_list):
        image = self.read_img(image_list)
        mask = self.read_img(mask_list, mask=True)
        return image, mask

    def get_dataset(self):
        """Loads data from the given config into a tf.data.Dataset and returns
        it."""
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.config['images'], self.config['labels'])
        )

        dataset = dataset.map(self._map_function,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self.augmentation.compose_sequential,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.batch(self.config['batch_size'],
                                drop_remainder=True)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
