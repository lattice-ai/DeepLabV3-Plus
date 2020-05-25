from glob import glob
from . import Dataset


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
