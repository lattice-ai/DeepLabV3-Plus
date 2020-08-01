import random
import tensorflow as tf


class Augmentation:

    def __init__(self, configs) -> None:
        self.configs = configs
        self.choice = None

    def apply_random_brightness(self, image, mask):
        condition = tf.cast(
            tf.random.uniform(
                [], maxval=2, dtype=tf.int32
            ), tf.bool
        )
        image = tf.cond(
            condition, lambda: tf.image.random_brightness(
            image, self.configs['random_brightness_max_delta']),
            lambda: tf.identity(image)
        )
        return image, mask
    
    def apply_random_contrast(self, image, mask):
        condition = tf.cast(
            tf.random.uniform(
                [], maxval=2, dtype=tf.int32
            ), tf.bool
        )
        image = tf.cond(
            condition, lambda: tf.image.random_contrast(
                image, self.configs['random_contrast_lower_bound'],
                self.configs['random_contrast_upper_bound']
            ), lambda: tf.identity(image)
        )
        return image, mask
    
    def apply_random_saturation(self, image, mask):
        condition = tf.cast(
            tf.random.uniform(
                [], maxval=2, dtype=tf.int32
            ), tf.bool
        )
        image = tf.cond(
            condition, lambda: tf.image.random_saturation(
            image, self.configs['random_contrast_lower_bound'],
                self.configs['random_contrast_upper_bound']
            ), lambda: tf.identity(image)
        )
        return image, mask
    
    def apply_horizontal_flip(self, image, mask):
        combined_tensor = tf.concat([image, mask], axis=2)
        combined_tensor = tf.image.random_flip_left_right(
            combined_tensor, seed=self.configs['seed']
        )
        image, mask = tf.split(
            combined_tensor,
            [self.configs['image_channels'], self.configs['label_channels']], axis=2
        )
        return image, mask
    
    def apply_vertical_flip(self, image, mask):
        combined_tensor = tf.concat([image, mask], axis=2)
        combined_tensor = tf.image.random_flip_up_down(
            combined_tensor, seed=self.configs['seed']
        )
        image, mask = tf.split(
            combined_tensor,
            [self.configs['image_channels'], self.configs['label_channels']], axis=2
        )
        return image, mask
    
    def apply_resize(self, image, mask):
        image = tf.image.resize(image, self.configs['image_size'])
        mask = tf.image.resize(mask, self.configs['image_size'], method="nearest")
        return image, mask
    
    def apply_random_crop(self, image, mask):
        condition = tf.cast(
            tf.random.uniform(
                [], maxval=2, dtype=tf.int32,
                seed=self.configs['seed']
            ), tf.bool
        )
        shape = tf.cast(tf.shape(image), tf.float32)
        h = tf.cast(shape[0] * self.configs['crop_percent'], tf.int32)
        w = tf.cast(shape[1] * self.configs['crop_percent'], tf.int32)
        combined_tensor = tf.concat([image, mask], axis=2)
        combined_tensor = tf.cond(
            condition, lambda: tf.image.random_crop(
                combined_tensor,
                [h, w, self.configs['image_channels'] + self.configs['label_channels']],
                seed=self.configs['seed']
            ), lambda: tf.identity(combined_tensor)
        )
        image, mask = tf.split(
            combined_tensor,
            [self.configs['image_channels'], self.configs['label_channels']], axis=2
        )
        return image, mask
    
    def compose_sequential(self, image, mask):
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
        self.choice = choice
    
    @tf.function
    def compose_by_choice(self, image, mask):

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
