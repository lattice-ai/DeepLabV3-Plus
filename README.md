# DeepLabV3-Plus (Ongoing)

[![](https://camo.githubusercontent.com/7ce7d8e78ad8ddab3bea83bb9b98128528bae110/68747470733a2f2f616c65656e34322e6769746875622e696f2f6261646765732f7372632f74656e736f72666c6f772e737667)](https://tensorflow.org/)

Tensorflow 2.2.0 implementation of DeepLabV3-Plus architecture as proposed by the paper [Encoder-Decoder with Atrous Separable
Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf).

![](./assets/deeplabv3_plus_diagram.png)

**Project Link:** [https://github.com/soumik12345/DeepLabV3-Plus/projects](https://github.com/soumik12345/DeepLabV3-Plus/projects).

**Experiments:** [https://app.wandb.ai/19soumik-rakshit96/deeplabv3-plus](https://app.wandb.ai/19soumik-rakshit96/deeplabv3-plus).

Model Architectures can be found [here](./models.md).

## Setup Datasets

- **Cityscapes**

    Register on [https://www.cityscapes-dataset.com/](https://www.cityscapes-dataset.com/).

    Edit `dataset/cityscapes.sh` and put in your username and password.

    ```shell script
    cd dataset
    bash cityscapes.sh
    ```

- **CamVid**
    
    Register on [https://www.kaggle.com/](https://www.kaggle.com/).
    
    Generate Kaggle API Token
    
    Edit `dataset/camvid.sh` and put in your kaggle username and kaggle key.
    
    ```shell script
    cd dataset
    bash camvid.sh
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
        'name': 'camvid',
        'train_image_list': sorted(glob('./dataset/CamVid/train/*')),
        'train_mask_list': sorted(glob('./dataset/CamVid/train_labels/*')),
        'val_image_list': sorted(glob('./dataset/CamVid/val/*')),
        'val_mask_list': sorted(glob('./dataset/CamVid/val_labels/*')),
        'patch_height': 512,
        'patch_width': 512,
        'train_batch_size': 8,
        'val_batch_size': 8
    },
    'wandb_project': 'deeplav-v3-plus',
    'strategy': tf.distribute.OneDeviceStrategy(device="/gpu:0"),
    'input_shape': (512, 512, 3),
    'backbone': 'resnet101',
    'n_classes': 32,
    'bn_momentum': 0.9997,
    'bn_epsilon': 1e-5,
    'learning_rate': 1e-4,
    'batch_size': 8,
    'epochs': 500,
    'weight_file': 'best_weights.h5'
}

trainer = Trainer(configurations)
history = trainer.train()
```

## Citation

```
@misc{1802.02611,
    Author = {Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam},
    Title = {Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
    Year = {2018},
    Eprint = {arXiv:1802.02611},
}
```