from glob import glob
from . import Dataset


class CamVidDataet(Dataset):

    def __init__(self, config):
        super(CamVidDataet, self).__init__(config)
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


if __name__ == '__main__':
    configurations = {
        'train_image_list': sorted(glob('../../dataset/CamVid/train/*')),
        'train_mask_list': sorted(glob('../../dataset/CamVid/train_labels/*')),
        'val_image_list': sorted(glob('../../dataset/CamVid/val/*')),
        'val_mask_list': sorted(glob('../../dataset/CamVid/val_labels/*')),
    }
    camvid_dataset = CamVidDataet(configurations)
