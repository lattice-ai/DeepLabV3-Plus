#!/usr/bin/env python
import os
from glob import glob

import tensorflow as tf

import wandb

from deeplabv3plus.train import Trainer


# Sample Configuration
config = {
    'wandb_api_key': 'xxxx-your_wandb_api_key-xxxx',
    'project_name': 'deeplabv3-plus',
    'experiment_name': 'camvid-segmentation-resnet-50-backbone',
    'train_dataset_configs': {
        'images': sorted(glob('./dataset/camvid/train/*')),
        'labels': sorted(glob('./dataset/camvid/trainannot/*')),
        'height': 360, 'width': 480, 'batch_size': 8
    },
    'val_dataset_configs': {
        'images': sorted(glob('./dataset/camvid/val/*')),
        'labels': sorted(glob('./dataset/camvid/valannot/*')),
        'height': 512, 'width': 512, 'batch_size': 8
    },
    'strategy': tf.distribute.OneDeviceStrategy(device="/gpu:0"),
    'num_classes': 20, 'height': 360, 'width': 480,
    'backbone': 'resnet50', 'learning_rate': 0.0001,
    # lambda for obtaining checkpoint lazily
    'checkpoint_path_getter': lambda: os.path.join(
        wandb.run.dir,
        'deeplabv3-plus-camvid-segmentation-resnet-50-backbone.h5'
    ),
    'epochs': 100
}

trainer = Trainer(config)
trainer.connect_wandb()
history = trainer.train()
