#!/usr/bin/env python

"""Module for training deeplabv3plus on camvid dataset."""

from glob import glob

import tensorflow as tf


# Sample Configuration
CONFIG = {
    'wandb_api_key': 'xxxx-your_wandb_api_key-xxxx',
    'project_name': 'deeplabv3-plus',
    'experiment_name': 'camvid-segmentation-resnet-50-backbone',

    'train_dataset_configs': {
        'images': sorted(glob('./dataset/camvid/train/*')),
        'labels': sorted(glob('./dataset/camvid/trainannot/*')),
        'height': 512, 'width': 512, 'batch_size': 8
    },

    'val_dataset_configs': {
        'images': sorted(glob('./dataset/camvid/val/*')),
        'labels': sorted(glob('./dataset/camvid/valannot/*')),
        'height': 512, 'width': 512, 'batch_size': 8
    },

    'strategy': tf.distribute.OneDeviceStrategy(device="/gpu:0"),
    'num_classes': 20, 'backbone': 'resnet50', 'learning_rate': 0.0001,

    'checkpoint_dir': "wandb://",
    'checkpoint_file_prefix': "deeplabv3plus_with_resnet50_",

    'epochs': 100
}
