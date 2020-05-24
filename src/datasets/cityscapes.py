from glob import glob
import tensorflow as tf


class CityscapesDataet:

    def __init__(self, config):
        self.config = config
        self.train_image_list = self.config['train_image_list']
        self.train_mask_list = self.config['train_mask_list']
        self.val_image_list = self.config['val_image_list']
        self.val_mask_list = self.config['val_mask_list']

    def assert_dataset(self):
        for i in range(len(self.train_image_list)):
            assert self.train_image_list[i].split(
                '/')[-1].split('_leftImg8bit')[0] == \
                   self.train_mask_list[i].split('/')[-1].split('_gtFine_labelIds')[0]
        print('Train Directories Good to go')
        for i in range(len(self.val_image_list)):
            assert self.val_image_list[i].split('/')[-1].split('_leftImg8bit')[0] == \
                   self.val_mask_list[i].split('/')[-1].split('_gtFine_labelIds')[0]
        print('Validation Directories Good to go')

    @staticmethod
    def read_image(image_path, img_height=800, img_width=1600, mask=False, flip=0):
        image = tf.io.read_file(image_path)
        if not mask:
            image = tf.cast(tf.image.decode_png(image, channels=3), dtype=tf.float32)
            image = tf.image.resize(images=image, size=[img_height, img_width])
            image = tf.image.random_brightness(image, max_delta=50.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.clip_by_value(image, 0, 255)
            image = tf.case([
                (tf.greater(flip, 0), lambda: tf.image.flip_left_right(image))
            ], default=lambda: image)
            image = image[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68])
        else:
            image = tf.image.decode_png(image, channels=1)
            image = tf.cast(tf.image.resize(images=image, size=[
                img_height, img_width]), dtype=tf.uint8)
            image = tf.case([
                (tf.greater(flip, 0), lambda: tf.image.flip_left_right(image))
            ], default=lambda: image)
        return image

    def random_crop(self, image, mask):
        image_dims = image.shape
        offset_h = tf.random.uniform(
            shape=(1,),
            maxval=image_dims[0] - self.config['patch_height'],
            dtype=tf.int32
        )[0]
        offset_w = tf.random.uniform(
            shape=(1,),
            maxval=image_dims[1] - self.config['patch_width'],
            dtype=tf.int32
        )[0]
        image = tf.image.crop_to_bounding_box(
            image, offset_height=offset_h,
            offset_width=offset_w,
            target_height=self.config['patch_height'],
            target_width=self.config['patch_width']
        )
        mask = tf.image.crop_to_bounding_box(
            mask, offset_height=offset_h,
            offset_width=offset_w,
            target_height=self.config['patch_height'],
            target_width=self.config['patch_width']
        )
        return image, mask

    def map_function(self, image_path, mask_path):
        flip = tf.random.uniform(
            shape=[1, ], minval=0, maxval=2, dtype=tf.int32)[0]
        image, mask = self.read_image(
            image_path, flip=flip
        ), self.read_image(
            mask_path, mask=True, flip=flip
        )
        image, mask = self.random_crop(image, mask)
        return image, mask
