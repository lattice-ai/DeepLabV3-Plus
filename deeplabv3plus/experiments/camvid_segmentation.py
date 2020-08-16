import os
import wandb
from glob import glob
from ..train import Trainer, tf

def run_training():
    config = {
        'wandb_api_key': 'kjbckajsbdksjbdkajsbkdasbkdj',
        'project_name': 'deeplabv3-plus',
        'experiment_name': 'human-parsing-resnet-50-backbone',
        'train_dataset_configs': {
            'images': sorted(glob('./camvid/train/*')),
            'labels': sorted(glob('./camvid/trainannot/*')),
            'height': 360, 'width': 480, 'batch_size': 8
        },
        'val_dataset_configs': {
            'images': sorted(glob('./camvid/val/*')),
            'labels': sorted(glob('./camvid/valannot/*')),
            'height': 512, 'width': 512, 'batch_size': 8
        },
        'strategy': tf.distribute.OneDeviceStrategy(device="/gpu:0"),
        'num_classes': 20, 'height': 360, 'width': 480,
        'backbone': 'resnet50', 'learning_rate': 0.0001,
        'checkpoint_path': os.path.join(
            wandb.run.dir,
            'deeplabv3-plus-human-parsing-resnet-50-backbone.h5'
        ),
        'epochs': 100
    }
    trainer = Trainer(config)
    trainer.connect_wandb()
    history = trainer.train()
    return history
