import os
import wandb
from glob import glob
import tensorflow as tf
from secret import WANDB_API_KEY
from src.model import DeeplabV3Plus
from wandb.keras import WandbCallback
from src.datasets.human_parsing import HumanParsingDataset


class Trainer:

    def __init__(self, configs):
        self.configs = configs
        self.assert_configs()

        # Train Dataset
        train_dataloader = HumanParsingDataset(self.configs['train_dataset_configs'])
        print('Data points in train dataset: {}'.format(len(train_dataloader)))
        self.train_dataset = train_dataloader.get_dataset()
        print('Train Dataset:', self.train_dataset)

        # Validation Dataset
        val_dataloader = HumanParsingDataset(self.configs['val_dataset_configs'])
        print('Data points in train dataset: {}'.format(len(val_dataloader)))
        self.val_dataset = val_dataloader.get_dataset()
        print('Val Dataset:', self.val_dataset)

        with self.configs['strategy'].scope():
            self.model = DeeplabV3Plus(
                self.configs['num_classes'], self.configs['height'],
                self.configs['width'], self.configs['backbone']
            )
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.configs['learning_rate']),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy']
            )

        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                self.configs['checkpoint_path'],
                monitor='loss', save_best_only=True, mode='min'
            ),
            WandbCallback()
        ]

    def assert_configs(self):
        assert 'project_name' in self.configs
        assert 'experiment_name' in self.configs
        assert 'train_dataset_configs' in self.configs
        assert 'val_dataset_configs' in self.configs
        assert 'strategy' in self.configs
        assert 'num_classes' in self.configs
        assert 'height' in self.configs
        assert 'width' in self.configs
        assert 'backbone' in self.configs
        assert 'learning_rate' in self.configs
        assert 'checkpoint_path' in self.configs
        assert 'epochs' in self.configs

    def connect_wandb(self):
        os.environ['WANDB_API_KEY'] = WANDB_API_KEY
        wandb.init(
            project=self.configs['project_name'],
            name=self.configs['experiment_name']
        )

    def train(self):
        history = self.model.fit(
            self.train_dataset, validation_data=self.val_dataset,
            steps_per_epoch=len(self.train_dataset) // self.configs['train_dataset_configs']['batch_size'],
            validation_steps=len(self.val_dataset) // self.configs['val_dataset_configs']['batch_size'],
            epochs=self.configs['epochs'], callbacks=self.callbacks
        )
        return history


if __name__ == '__main__':
    config = {
        'project_name': 'deeplabv3-plus',
        'experiment_name': 'human-parsing-resnet-50-backbone',
        'train_dataset_configs': {
            'images': sorted(glob(
                './dataset/instance-level_human_parsing/instance-level_human_parsing/Training/Images/*'
            )),
            'labels': sorted(glob(
                './dataset/instance-level_human_parsing/instance-level_human_parsing/Training/Category_ids/*'
            )),
            'height': 512, 'width': 512, 'batch_size': 8
        },
        'val_dataset_configs': {
            'images': sorted(glob(
                './dataset/instance-level_human_parsing/instance-level_human_parsing/Validation/Images/*'
            )),
            'labels': sorted(glob(
                './dataset/instance-level_human_parsing/instance-level_human_parsing/Validation/Category_ids/*'
            )),
            'height': 512, 'width': 512, 'batch_size': 8
        },
        'strategy': tf.distribute.OneDeviceStrategy(device="/gpu:0"),
        'num_classes': 20, 'height': 512, 'width': 512,
        'backbone': 'resnet50', 'learning_rate': 0.0001,
        'checkpoint_path': 'deeplabv3-plus-human-parsing-resnet-50-backbone.h5',
        'epochs': 100
    }
    trainer = Trainer(config)
