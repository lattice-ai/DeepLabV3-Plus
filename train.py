import tensorflow as tf
from src.model import DeepLabV3Plus
from src.datasets.cityscapes import CityscapesDataet


class Trainer:

    def __init__(self, config):
        self.config = config
        cityscapes_dataset = CityscapesDataet(self.config['dataset_config'])
        self.train_dataset, self.val_dataset = cityscapes_dataset.get_datasets()
        self.strategy = self.config['strategy']
        self.model = self.define_model()

    def define_model(self):
        with self.strategy.scope():
            model = DeepLabV3Plus(
                input_shape=self.config['input_shape'],
                backbone=self.config['backbone'],
                n_classes=self.config['n_classes']
            )
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.momentum = 0.9997
                    layer.epsilon = 1e-5
                elif isinstance(layer, tf.keras.layers.Conv2D):
                    layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)
            model.compile(
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.optimizers.Adam(learning_rate=1e-4), metrics=['accuracy']
            )
        return model
