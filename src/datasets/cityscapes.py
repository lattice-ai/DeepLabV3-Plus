from glob import glob
from . import Dataset
import tensorflow as tf


class CityscapesDataet(Dataset):

    def __init__(self, config):
        super(CityscapesDataet, self).__init__(config)
        self.assert_dataset()

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

    def map_function(self, image_path, mask_path):
        flip = tf.random.uniform(
            shape=[1, ], minval=0, maxval=2, dtype=tf.int32)[0]
        image, mask = self.read_image(
            image_path, flip=flip,
            img_height=800, img_width=1600
        ), self.read_image(
            mask_path, mask=True, flip=flip,
            img_height=800, img_width=1600
        )
        image, mask = self.random_crop(image, mask)
        return image, mask


if __name__ == '__main__':
    configurations = {
        'train_image_list': sorted(glob('../../datasets/cityscapes/dataset/train_images/*')),
        'train_mask_list': sorted(glob('../../datasets/cityscapes/dataset/train_masks/*')),
        'val_image_list': sorted(glob('../../datasets/cityscapes/dataset/val_images/*')),
        'val_mask_list': sorted(glob('../../datasets/cityscapes/dataset/val_masks/*')),
        'patch_height': 512,
        'patch_width': 512,
        'train_batch_size': 16,
        'val_batch_size': 16
    }
    cityscapes_dataset = CityscapesDataet(configurations)
