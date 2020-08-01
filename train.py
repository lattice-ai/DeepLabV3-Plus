import os
import wandb
from glob import glob
import tensorflow as tf
from wandb.keras import WandbCallback
from src.datasets import GenericDataset
from src.model.deeplabv3_plus import DeeplabV3Plus
from src.datasets.augmentations import Augmentation


class Trainer:

    def __init__(self, configs):
        self.configs = configs
        self.assert_configs()

        # Train Dataset
        train_dataloader = GenericDataset(self.configs['train_dataset_configs'])
        self.train_data_length = len(train_dataloader)
        print('Data points in train dataset: {}'.format(self.train_data_length))
        self.train_dataset = train_dataloader.get_dataset()
        print('Train Dataset:', self.train_dataset)

        # Validation Dataset
        val_dataloader = GenericDataset(self.configs['val_dataset_configs'])
        self.val_data_length = len(val_dataloader)
        print('Data points in train dataset: {}'.format(self.val_data_length))
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

    def assert_configs(self):
        assert 'wandb_api_key' in self.configs
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
        os.environ['WANDB_API_KEY'] = self.configs['wandb_api_key']
        wandb.init(
            project=self.configs['project_name'],
            name=self.configs['experiment_name']
        )

    def train(self):
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                self.configs['checkpoint_path'],
                monitor='loss', save_best_only=True, mode='min'
            )
        ]
        try:
            self.callbacks.append(WandbCallback())
        except:
            self.callbacks.append(
                tf.keras.callbacks.TensorBoard(log_dir='logs')
            )
        history = self.model.fit(
            self.train_dataset, validation_data=self.val_dataset,
            steps_per_epoch=self.train_data_length // self.configs['train_dataset_configs']['batch_size'],
            validation_steps=self.val_data_length // self.configs['val_dataset_configs']['batch_size'],
            epochs=self.configs['epochs'], callbacks=self.callbacks
        )
        return history


if __name__ == '__main__':
    augmentation = Augmentation(
        configs={
            'random_brightness_max_delta': 0.1,
            'random_contrast_lower_bound': 0.1,
            'random_contrast_upper_bound': 0.8,
            'random_saturation_lower_bound': 0.1,
            'random_saturation_upper_bound': 0.8,
            'seed': 47, 'image_channels': 3, 'label_channels': 1,
            'image_size': (512, 512), 'crop_percent': 0.9,
        }
    )
    config = {
        'wandb_api_key': 'kjbckajsbdksjbdkajsbkdasbkdj',
        'project_name': 'deeplabv3-plus',
        'experiment_name': 'human-parsing-resnet-50-backbone',
        'train_dataset_configs': {
            'images': sorted(glob(
                './dataset/CamVid/train/*'
            )),
            'labels': sorted(glob(
                './dataset/CamVid/train_labels/*'
            )),
            'batch_size': 8,
            'augment_compose_function': augmentation.compose_sequential
        },
        'val_dataset_configs': {
            'images': sorted(glob(
                './dataset/CamVid/val/*'
            )),
            'labels': sorted(glob(
                './dataset/CamVid/val_labels/*'
            )),
            'batch_size': 8
        },
        'strategy': tf.distribute.OneDeviceStrategy(device="/gpu:0"),
        'num_classes': 20, 'height': 512, 'width': 512,
        'backbone': 'resnet50', 'learning_rate': 0.0001,
        'checkpoint_path': 'deeplabv3-plus-human-parsing-resnet-50-backbone.h5',
        'epochs': 100
    }
    trainer = Trainer(config)
    trainer.connect_wandb()
    history = trainer.train()
