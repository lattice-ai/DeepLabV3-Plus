"""Module providing Trainer class for deeplabv3plus"""

import os

import tensorflow as tf

import wandb
from wandb.keras import WandbCallback

from deeplabv3plus.datasets import GenericDataLoader
from deeplabv3plus.model import DeeplabV3Plus


class Trainer:
    """Class for managing DeeplabV3+ model training.

    Args:
        configs:
            python dictionary containing training configuration for
            DeeplabV3Plus
    """
    def __init__(self, configs):
        self.configs = configs
        self._assert_configs()

        # Train Dataset
        train_dataloader = GenericDataLoader(self.configs[
            'train_dataset_configs'])
        self.train_data_length = len(train_dataloader)
        print('[+] Data points in train dataset: {}'.format(
            self.train_data_length))
        self.train_dataset = train_dataloader.get_dataset()
        print('Train Dataset:', self.train_dataset)

        # Validation Dataset
        val_dataloader = GenericDataLoader(self.configs[
            'val_dataset_configs'])
        self.val_data_length = len(val_dataloader)
        print('Data points in train dataset: {}'.format(
            self.val_data_length))
        self.val_dataset = val_dataloader.get_dataset()
        print('Val Dataset:', self.val_dataset)

        self._model = None
        self._wandb_initialized = False

    @property
    def model(self):
        """Property returning model being trained."""

        if self._model is not None:
            return self._model

        with self.configs['strategy'].scope():
            self._model = DeeplabV3Plus(
                num_classes=self.configs['num_classes'],
                backbone=self.configs['backbone']
            )

            self._model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.configs['learning_rate']
                ),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy']
            )

            return self._model

    @staticmethod
    def _assert_dataset_config(dataset_config):
        assert 'images' in dataset_config and \
            isinstance(dataset_config['images'], list)
        assert 'labels' in dataset_config and \
            isinstance(dataset_config['labels'], list)

        assert 'height' in dataset_config and \
            isinstance(dataset_config['height'], int)
        assert 'width' in dataset_config and \
            isinstance(dataset_config['width'], int)

        assert 'batch_size' in dataset_config and \
            isinstance(dataset_config['batch_size'], int)

    def _assert_configs(self):
        assert 'project_name' in self.configs and \
            isinstance(self.configs['project_name'], str)
        assert 'experiment_name' in self.configs and \
            isinstance(self.configs['experiment_name'], str)

        assert 'train_dataset_configs' in self.configs
        Trainer._assert_dataset_config(self.configs['train_dataset_configs'])
        assert 'val_dataset_configs' in self.configs
        Trainer._assert_dataset_config(self.configs['val_dataset_configs'])

        assert 'strategy' in self.configs and \
            isinstance(self.configs['strategy'], tf.distribute.Strategy)

        assert 'num_classes' in self.configs and \
            isinstance(self.configs['num_classes'], int)
        assert 'backbone' in self.configs and \
            isinstance(self.configs['backbone'], str)

        assert 'learning_rate' in self.configs and \
            isinstance(self.configs['learning_rate'], float)

        assert 'checkpoint_dir' in self.configs and \
            isinstance(self.configs['checkpoint_dir'], str)
        assert 'checkpoint_file_prefix' in self.configs and \
            isinstance(self.configs['checkpoint_file_prefix'], str)

        assert 'epochs' in self.configs and \
            isinstance(self.configs['epochs'], int)

    def connect_wandb(self):
        """Connects Trainer to wandb.

        Runs wandb.init() with the given wandb_api_key, project_name and
        experiment_name.
        """
        if 'wandb_api_key' not in self.configs:
            return

        os.environ['WANDB_API_KEY'] = self.configs['wandb_api_key']
        wandb.init(
            project=self.configs['project_name'],
            name=self.configs['experiment_name']
        )
        self._wandb_initialized = True

    def _get_checkpoint_filename_format(self):
        if self.configs['checkpoint_dir'] == 'wandb://':
            if 'wandb_api_key' not in self.configs:
                raise ValueError("Invalid configuration, wandb_api_key not "
                                 "provided!")
            if not self._wandb_initialized:
                raise ValueError("Wandb not intialized, "
                                 "checkpoint_filename_format is unusable.")

            return os.path.join(wandb.run.dir,
                                self.configs['checkpoint_file_prefix'] +
                                "{epoch}")

        return os.path.join(self.configs['checkpoint_dir'],
                            self.configs['checkpoint_file_prefix'] +
                            "{epoch}")

    def _get_logger_callback(self):
        if 'wandb_api_key' not in self.configs:
            return tf.keras.callbacks.TensorBoard()

        try:
            return WandbCallback()
        except wandb.Error as error:
            if 'wandb_api_key' in self.configs:
                raise error  # rethrow

            print("[-] Defaulting to TensorBoard logging...")
            return tf.keras.callbacks.TensorBoard()

    def train(self):
        """Trainer entry point.

        Attempts to connect to wandb before starting training. Runs .fit() on
        loaded model.
        """
        if not self._wandb_initialized:
            self.connect_wandb()

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self._get_checkpoint_filename_format(),
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                save_weights_only=False
            ),

            self._get_logger_callback()
        ]

        history = self.model.fit(
            self.train_dataset, validation_data=self.val_dataset,

            steps_per_epoch=self.train_data_length //
            self.configs['train_dataset_configs']['batch_size'],

            validation_steps=self.val_data_length //
            self.configs['val_dataset_configs']['batch_size'],

            epochs=self.configs['epochs'], callbacks=callbacks
        )

        return history
