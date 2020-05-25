from glob import glob
from . import Dataset
import tensorflow as tf


class CamVidDataset(Dataset):

    def __init__(self, config):
        super(CamVidDataset, self).__init__(config)
        self.assert_dataset()

    def assert_dataset(self):
        for i in range(len(self.train_image_list)):
            assert self.train_image_list[i].split('/')[-1][:-4] == \
                   self.train_mask_list[i].split('/')[-1][:-6]
        print('Train Directories Good to go')
        for i in range(len(self.val_image_list)):
            assert self.val_image_list[i].split('/')[-1][:-4] == \
                   self.val_mask_list[i].split('/')[-1][:-6]
        print('Validation Directories Good to go')

    def map_function(self, image_path, mask_path):
        flip = tf.random.uniform(
            shape=[1, ], minval=0, maxval=2, dtype=tf.int32)[0]
        image, mask = self.read_image(
            image_path, flip=flip,
            img_height=960, img_width=720
        ), self.read_image(
            mask_path, mask=True, flip=flip,
            img_height=960, img_width=720
        )
        image, mask = self.random_crop(image, mask)
        return image, mask


if __name__ == '__main__':
    configurations = {
        'train_image_list': sorted(glob('../../dataset/CamVid/train/*')),
        'train_mask_list': sorted(glob('../../dataset/CamVid/train_labels/*')),
        'val_image_list': sorted(glob('../../dataset/CamVid/val/*')),
        'val_mask_list': sorted(glob('../../dataset/CamVid/val_labels/*')),
    }
    camvid_dataset = CamVidDataset(configurations)
