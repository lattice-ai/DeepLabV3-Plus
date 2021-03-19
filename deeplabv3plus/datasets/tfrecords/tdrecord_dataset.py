"""
Module providing a class for representing TF Record based datasets.
"""

from typing import List
import tensorflow as tf

from .tfrecord_loader import TFRecordLoader
from .commons import plot_result


class TFRecordDataset:
    """
    Wrapper class for wrapping tf.data.Dataset instances. Builds
    tf.data.Dataset from a list of tf records.

    Args:
        tfrecords: List of tfrecord str representations
    """

    def __init__(self, tfrecords: List[str], image_size: int):
        self._tfrecords = tfrecords
        self._image_size = image_size
        self._dataset = None

    @property
    def dataset(self) -> tf.data.Dataset:
        """
        Loads dataset from tfrecords and returns a
        preconfigured instance of tf.data.Dataset
        
        Returns:
            instance of tf.data.Dataset
        """
        if self._dataset is not None:
            return self._dataset

        loader = TFRecordLoader(self._image_size)
        self._dataset = configure_dataset(
            loader.get_dataset(self._tfrecords))

        return self._dataset

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
