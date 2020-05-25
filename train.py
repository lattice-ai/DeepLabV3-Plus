import os
import wandb as wb
from glob import glob
import tensorflow as tf
from src.model import DeepLabV3Plus
from src.datasets.cityscapes import CityscapesDataet


class Trainer:

    def __init__(self, config):
        self.config = config
        wb.init(project=self.config['wandb_project'])
        self.strategy = self.config['strategy']
        with self.strategy.scope():
            self.cityscapes_dataset = CityscapesDataet(self.config['dataset_config'])
            self.train_dataset, self.val_dataset = self.cityscapes_dataset.get_datasets()
            self.model = self.define_model()

    def define_model(self):
        model = DeepLabV3Plus(
            input_shape=self.config['input_shape'],
            backbone=self.config['backbone'],
            n_classes=self.config['n_classes']
        )
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.momentum = self.config['bn_momentum'],
                layer.epsilon = self.config['bn_epsilon']
            elif isinstance(layer, tf.keras.layers.Conv2D):
                layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.optimizers.Adam(learning_rate=1e-4), metrics=['accuracy']
        )
        return model

    def train(self):
        train_steps_per_epoch = len(
            self.cityscapes_dataset.train_image_list
        ) // self.config['batch_size']
        val_steps_per_epoch = len(
            self.cityscapes_dataset.val_image_list
        ) // self.config['batch_size']
        history = self.model.fit(
            self.train_dataset,
            steps_per_epoch=train_steps_per_epoch,
            epochs=self.config['epochs'],
            validation_data=self.val_dataset,
            validation_steps=val_steps_per_epoch,
            callbacks=[
                wb.keras.WandbCallback(),
                tf.keras.callbacks.ModelCheckpoint(
                    mode='min', filepath=self.config['weight_file'],
                    monitor='val_loss', save_best_only='True',
                    save_weights_only='True', verbose=1
                )
            ]
        )
        self.model.save(os.path.join(wb.run.dir, self.config['weight_file']))
        return history


if __name__ == '__main__':
    configurations = {
        'dataset_config': {
            'train_image_list': sorted(glob('cityscapes/dataset/train_images/*')),
            'train_mask_list': sorted(glob('cityscapes/dataset/train_masks/*')),
            'val_image_list': sorted(glob('cityscapes/dataset/val_images/*')),
            'val_mask_list': sorted(glob('cityscapes/dataset/val_masks/*')),
            'patch_height': 512,
            'patch_width': 512,
            'train_batch_size': 16,
            'val_batch_size': 16
        },
        'wandb_project': 'deeplav-v3-plus',
        'strategy': tf.distribute.OneDeviceStrategy(device="/gpu:0"),
        'input_shape': (512, 512, 3),
        'backbone': 'resnet101',
        'n_classes': 66,
        'bn_momentum': 0.9997,
        'bn_epsilon': 1e-5,
        'batch_size': 16,
        'epochs': 500,
    }
    trainer = Trainer(configurations)
    history = trainer.train()
