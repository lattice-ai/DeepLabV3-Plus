# DeepLabV3-Plus (Ongoing)

Tensorflow 2.2.0 implementation of DeepLabV3-Plus architecture as proposed by the paper [Encoder-Decoder with Atrous Separable
Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf).

![](./assets/deeplabv3_plus_diagram.png)

**Project Link:** [https://github.com/soumik12345/DeepLabV3-Plus/projects](https://github.com/soumik12345/DeepLabV3-Plus/projects).

Model Architectures can be found [here](./models.md).

## Setup Cityscapes

Register on [https://www.cityscapes-dataset.com/](https://www.cityscapes-dataset.com/).

Edit `dataset/cityscapes.sh` and put in your username and password.

```shell script
cd dataset
bash cityscapes.sh
```

## Code to test Model

```python
from src.model import DeepLabV3Plus

model = DeepLabV3Plus(backbone='resnet101')
model.summary()
```

## Training

```shell script
wandb login <login-key>
python3 train.py
```

If you want to use a custom configuration, you can define it in the following way:

```python
from glob import glob
import tensorflow as tf
from train import Trainer

# Sample Configuration
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
    'strategy': tf.distribute.OneDeviceStrategy(),
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
```