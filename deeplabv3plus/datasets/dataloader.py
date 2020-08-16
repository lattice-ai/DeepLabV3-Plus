import tensorflow as tf


class GenericDataLoader:

    def __init__(self, configs):
        self.configs = configs
        self.assert_dataset()

    def assert_dataset(self):
        assert 'images' in self.configs and 'labels' in self.configs
        assert len(self.configs['images']) == len(self.configs['labels'])
        print('Train Images are good to go')

    def __len__(self):
        return len(self.configs['images'])

    def read_img(self, image_path, mask=False):
        image = tf.io.read_file(image_path)
        if mask:
            image = tf.image.decode_png(image, channels=1)
            image.set_shape([None, None, 1])
            image = (tf.image.resize(
                images=image, size=[
                    self.configs['height'],
                    self.configs['width']
                ], method="nearest"
            ))
            image = tf.cast(image, tf.float32)
        else:
            image = tf.image.decode_png(image, channels=3)
            image.set_shape([None, None, 3])
            image = (tf.image.resize(
                images=image, size=[
                    self.configs['height'],
                    self.configs['width']
                ]
            ))
            image = tf.cast(image, tf.float32) / 127.5 - 1
        return image

    def _map_function(self, image_list, mask_list):
        image = self.read_img(image_list)
        mask = self.read_img(mask_list, mask=True)
        return image, mask

    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.configs['images'], self.configs['labels'])
        )
        dataset = dataset.map(self._map_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.configs['batch_size'], drop_remainder=True)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
