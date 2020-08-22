"""Module to provide implemtation of various image augmentation operations."""

import random
import tensorflow as tf


class Augmentation:
    """Class to provide augmentation mapper functions for dataset.
    Args:
        config:
            dictionary containing configuration information
    """
    def __init__(self, config) -> None:
        self.config = config
        assert 'random_brightness_max_delta' in self.config
        self.choice = None

    def apply_random_brightness(self, image, mask):
        """Applies random brightness.

        Args:
            image:
                image to apply brightness on
            mask:
                corresponding mask for the given image, which remains
                unchanged
        """
        condition = tf.cast(
            tf.random.uniform(
                [], maxval=2, dtype=tf.int32
            ), tf.bool
        )
        image = tf.cond(
            condition,
            lambda: tf.image.random_brightness(
                image, self.config['random_brightness_max_delta']
            ),
            lambda: tf.identity(image)
        )
        return image, mask

    def apply_random_contrast(self, image, mask):
        """Applies random contrast.

        Args:
            image:
                image to apply random contrast on
            mask:
                corresponding mask for the given image, which remains
                unchanged.
        """
        condition = tf.cast(
            tf.random.uniform(
                [], maxval=2, dtype=tf.int32
            ), tf.bool
        )
        image = tf.cond(
            condition,
            lambda: tf.image.random_contrast(
                image, self.config['random_contrast_lower_bound'],
                self.config['random_contrast_upper_bound']
            ),
            lambda: tf.identity(image)
        )
        return image, mask

    def apply_random_saturation(self, image, mask):
        """Applies random saturation.

        Args:
            image:
                image to apply random saturation on
            mask:
                corresponding mask for the given image, which remains
                unchanged.
        """
        condition = tf.cast(
            tf.random.uniform(
                [], maxval=2, dtype=tf.int32
            ), tf.bool
        )
        image = tf.cond(
            condition,
            lambda: tf.image.random_saturation(
                image, self.config['random_contrast_lower_bound'],
                self.config['random_contrast_upper_bound']
            ),
            lambda: tf.identity(image)
        )
        return image, mask

    def apply_horizontal_flip(self, image, mask):
        """Flips both the image and mask horizontally.

        Args:
            image:
                image to horizontally flip
            mask:
                corresponding mask for the given image, which is flipped along
                with the image.
        """
        combined_tensor = tf.concat([image, mask], axis=2)
        combined_tensor = tf.image.random_flip_left_right(
            combined_tensor, seed=self.config['seed']
        )
        image, mask = tf.split(
            combined_tensor,
            [self.config['image_channels'],
             self.config['label_channels']],
            axis=2
        )
        return image, mask

    def apply_vertical_flip(self, image, mask):
        """Flips both the image and mask vertically.

        Args:
            image:
                image to flip vertically
            mask:
                corresponding mask for the given image, which is flipped along
                with the image.
        """
        combined_tensor = tf.concat([image, mask], axis=2)
        combined_tensor = tf.image.random_flip_up_down(
            combined_tensor, seed=self.config['seed']
        )
        image, mask = tf.split(
            combined_tensor,
            [self.config['image_channels'],
             self.config['label_channels']],
            axis=2
        )
        return image, mask

    def apply_resize(self, image, mask):
        """Resizes both the image and mask.

        Args:
            image:
                image to resize
            mask:
                corresponding mask for the given image, which is resized along
                with the image.
        """
        image = tf.image.resize(image,
                                (self.config['target_image_height'],
                                 self.config['target_image_width']))
        mask = tf.image.resize(mask,
                               (self.config['target_image_height'],
                                self.config['target_image_width']),
                               method="nearest")
        return image, mask

    def apply_random_crop(self, image, mask):
        """Applies random crop to both the image and mask.

        In order to maintain consistency between the mask and the image, we
        stack them along the channel dimension, and apply random crop on the
        stacked image.

        Args:
            image:
                image to randomly crop.
            mask:
                corresponding mask for the given image, which is cropped at
                exactly the same offsets, with the same dimenstions as the
                original image.
        """
        condition = tf.cast(
            tf.random.uniform(
                [], maxval=2, dtype=tf.int32,
                seed=self.config['seed']
            ), tf.bool
        )
        shape = tf.cast(tf.shape(image), tf.float32)

        # !pylint:disable=invalid-name
        h = tf.cast(shape[0] * self.config['crop_percent'], tf.int32)
        w = tf.cast(shape[1] * self.config['crop_percent'], tf.int32)
        combined_tensor = tf.concat([image, mask], axis=2)
        combined_tensor = tf.cond(
            condition, lambda: tf.image.random_crop(
                combined_tensor,
                [h, w, self.config['image_channels'] +
                 self.config['label_channels']],
                seed=self.config['seed']
            ), lambda: tf.identity(combined_tensor)
        )
        image, mask = tf.split(
            combined_tensor,
            [self.config['image_channels'],
             self.config['label_channels']],
            axis=2
        )
        return image, mask

    def compose_sequential(self, image, mask):
        """Applies all augmentation operations (as implemented in this class)
        sequentially on the given image, mask pair.

        Args:
            image, mask:
                image mask pair to apply operations on.
        """
        image, mask = self.apply_random_brightness(image, mask)
        image, mask = self.apply_random_contrast(image, mask)
        image, mask = self.apply_random_saturation(image, mask)
        image, mask = self.apply_random_crop(image, mask)
        image, mask = self.apply_horizontal_flip(image, mask)
        image, mask = self.apply_vertical_flip(image, mask)
        image, mask = self.apply_resize(image, mask)
        return image, mask

    @tf.function
    def compose_random(self, image, mask):
        """Applies augmentation operations in random order.

        Args:
            image, mask:
                image, mask pair to apply augmentatons on.
        """
        def compose(_image, _mask):
            options = [
                self.apply_random_brightness,
                self.apply_random_contrast,
                self.apply_random_saturation,
                self.apply_random_crop,
                self.apply_horizontal_flip,
                self.apply_vertical_flip
            ]
            augment_func = random.choice(options)
            _image, _mask = augment_func(_image, _mask)
            _image, _mask = self.apply_resize(_image, _mask)
            return _image, _mask

        return tf.py_function(
            compose, [image, mask],
            [tf.float32, tf.uint8]
        )

    def set_choice(self, choice):
        """Sets choice for random compose by choice augmentation.

        Args:
            choice:
                choice to set for random.choice
        """
        self.choice = choice

    @tf.function
    def compose_by_choice(self, image, mask):
        """Applies random compose by choice operation using the choice set.

        Args:
            image, mask:
                image and mask to apply augmentation on.
        """

        def compose(_image, _mask):
            options = self.choice
            augment_func = random.choice(options)
            _image, _mask = augment_func(_image, _mask)
            _image, _mask = self.apply_resize(_image, _mask)
            return _image, _mask

        return tf.py_function(
            compose, [image, mask],
            [tf.float32, tf.uint8]
        )
