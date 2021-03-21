"""
Module providing a class for representing TF Record based datasets.
"""

from typing import List

import tensorflow as tf
import numpy as np

from .tfrecord_loader import TFRecordLoader
from .commons import plot_result
from .augmentations import AugmentationFactory


class TFRecordDataset:
    """
    Wrapper class for wrapping tf.data.Dataset instances. Builds
    tf.data.Dataset from a list of tf records.

    Args:
        tfrecords: List of tfrecord str representations
    """

    def __init__(self,
                 tfrecords: List[str],
                 image_size: int,
                 apply_flips: bool,
                 apply_jitter: bool):

        self._tfrecords = tfrecords
        self._image_size = image_size
        self._dataset = None

        self._apply_flips = apply_flips
        self._apply_jitter = apply_jitter

    def create_dataset(self):
        if self._dataset is not None:
            return self._dataset

        loader = TFRecordLoader(self._image_size)
        self._dataset = loader.get_dataset(self._tfrecords)

        augmentation_factory = AugmentationFactory(
            apply_horizontal_flip=self._apply_flips,
            apply_jitter=self._apply_jitter
        )

        self._dataset = augmentation_factory.augment_dataset(
            self._dataset)

    def summary(self, visualize: bool = False, num_samples: int = 4):
        """
        Logs a summary of the loaded dataset instance. Optionally
        visualized some sample images from the dataset.

        Args:
            visualize:
                bool - whether or not to visualize
            num_samples:
                int - no of samples to visualize if visualizing
        """
        print(self._dataset)

        if not visualize:
            return

        for x, y in self._dataset.take(num_samples):
            x = (x + 1) * 127.5

            plot_result([x.numpy().astype(np.uint8),
                         y.numpy().astype(np.uint8)],
                        ['Image', 'Label'], (20, 6))

    def configured_dataset(
            self,
            shuffle_buffer: int = 127,
            batch_size: int = 16):

        __dataset = self._dataset.repeat()
        __dataset = __dataset.shuffle(shuffle_buffer)
        __dataset = __dataset.batch(batch_size)
        __dataset = __dataset.prefetch(tf.data.AUTOTUNE)
        return __dataset
