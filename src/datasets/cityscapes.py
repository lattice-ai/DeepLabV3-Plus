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
