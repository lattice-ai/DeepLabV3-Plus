import tensorflow as tf


class HumanParsingDataset:

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
        img = tf.io.read_file(image_path)
        if mask:
            img = tf.image.decode_png(img, channels=1)
            img.set_shape([None, None, 1])
            img = (tf.image.resize(
                images=img, size=[
                    self.configs['height'],
                    self.configs['width']
                ]
            ))
            img = tf.cast(img, tf.float32)
        else:
            img = tf.image.decode_png(img, channels=3)
            img.set_shape([None, None, 3])
            img = (tf.image.resize(
                images=img, size=[
                    self.configs['height'],
                    self.configs['width']
                ]
            ))
            img = tf.cast(img, tf.float32) / 127.5 - 1
        return img

    def _map_function(self, img_list, mask_list):
        img = self.read_img(img_list)
        mask = self.read_img(mask_list, mask=True)
        return img, mask

    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.configs['images'], self.configs['labels'])
        )
        dataset = dataset.map(self._map_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.configs['batch_size'], drop_remainder=True)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
