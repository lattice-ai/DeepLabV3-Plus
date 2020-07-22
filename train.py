import os
import wandb
import tensorflow as tf
from secret import WANDB_API_KEY
from src.model import DeeplabV3Plus
from wandb.keras import WandbCallback
from src.datasets.human_parsing import HumanParsingDataset


class Trainer:

    def __init__(self, configs):
        self.configs = configs

        # Wandb Init
        os.environ['WANDB_API_KEY'] = WANDB_API_KEY
        wandb.init(
            project=self.configs['project_name'],
            name=self.configs['experiment_name']
        )

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

    def train(self):
        history = self.model.fit(
            self.train_dataset, validation_data=self.val_dataset,
            steps_per_epoch=len(self.val_dataset) // self.configs['batch_size'],
            validation_steps=len(self.val_dataset) // self.configs['batch_size'],
            epochs=self.configs['epochs'], callbacks=self.callbacks
        )
        return history
