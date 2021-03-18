import tensorflow as tf


class AugmentationFactory:

    def __init__(self, apply_horizontal_flip: bool, apply_jitter: bool):
        self.apply_horizontal_flip = apply_horizontal_flip
        self.apply_jitter = apply_jitter
        self.image_channels, self.label_channels = 3, 1

    def _flip_horizontal(self, image, label):
        comb_tensor = tf.concat([image, label], axis=2)
        comb_tensor = tf.image.random_flip_left_right(comb_tensor)
        image, label = tf.split(comb_tensor, [self.image_channels, self.label_channels], axis=-1)
        return image, label

    @staticmethod
    def _random_jitter(image):
        image = tf.image.stateless_random_saturation(image, 0.9, 1.1)
        image = tf.image.stateless_random_brightness(image, 0.075)
        image = tf.image.stateless_random_contrast(image, 0.9, 1.1)
        return image

    def _map_augmentations(self, image, label):
        if self.apply_horizontal_flip:
            image = self._flip_horizontal(image=image, label=label)
        image = self._random_jitter(image=image, seed=seed) if self.apply_jitter else image
        return image, label

    def augment_dataset(self, dataset):
        return dataset.map(
            map_func=self._map_augmentations,
            num_parallel_calls=tf.data.AUTOTUNE
        )
