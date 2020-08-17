import os
import wandb
import tensorflow as tf
from wandb.keras import WandbCallback
from .datasets import GenericDataLoader
from .model import DeeplabV3Plus


class Trainer:

    def __init__(self, configs):
        self.configs = configs
        self.assert_configs()

        # Train Dataset
        train_dataloader = GenericDataLoader(self.configs['train_dataset_configs'])
        self.train_data_length = len(train_dataloader)
        print('Data points in train dataset: {}'.format(self.train_data_length))
        self.train_dataset = train_dataloader.get_dataset()
        print('Train Dataset:', self.train_dataset)

        # Validation Dataset
        val_dataloader = GenericDataLoader(self.configs['val_dataset_configs'])
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
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                self.configs['checkpoint_path'],
                monitor='val_loss', save_best_only=True,
                mode='min', save_weights_only=True
            )
        ]
        try:
            callbacks.append(WandbCallback())
        except:
            callbacks.append(
                tf.keras.callbacks.TensorBoard(log_dir='logs')
            )
        history = self.model.fit(
            self.train_dataset, validation_data=self.val_dataset,
            steps_per_epoch=self.train_data_length // self.configs['train_dataset_configs']['batch_size'],
            validation_steps=self.val_data_length // self.configs['val_dataset_configs']['batch_size'],
            epochs=self.configs['epochs'], callbacks=callbacks
        )
        return history
