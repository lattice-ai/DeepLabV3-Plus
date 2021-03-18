import tensorflow as tf


class AugmentationFactory:

    def __init__(self, apply_horizontal_flip: bool, apply_vertical_flip: bool):
        self.apply_horizontal_flip = apply_horizontal_flip
        self.apply_vertical_flip = apply_vertical_flip
        self.image_channels, self.label_channels = 3, 1

    def _flip_horizontal(self, image, label, seed):
        comb_tensor = tf.concat([image, label], axis=2)
        comb_tensor = tf.image.random_flip_left_right(comb_tensor, seed=seed)
        image, label = tf.split(comb_tensor, [self.image_channels, self.label_channels], axis=-1)
        return image, label

    def _flip_vertical(self, image, label, seed):
        comb_tensor = tf.concat([image, label], axis=2)
        comb_tensor = tf.image.random_flip_up_down(comb_tensor, seed=seed)
        image, label = tf.split(comb_tensor, [self.image_channels, self.label_channels], axis=-1)
        return image, label

    @staticmethod
    def _random_jitter(image, seed):
        image = tf.image.stateless_random_saturation(image, 0.9, 1.1, seed)
        image = tf.image.stateless_random_brightness(image, 0.075, seed)
        image = tf.image.stateless_random_contrast(image, 0.9, 1.1, seed)
        return image
