# DeepLabV3-Plus (Ongoing)

Tensorflow 2.2.0 implementation of DeepLabV3-Plus architecture as proposed by the paper [Encoder-Decoder with Atrous Separable
Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf).

![](./assets/deeplabv3_plus_diagram.png)

**Project Link:** [https://github.com/soumik12345/DeepLabV3-Plus/projects](https://github.com/soumik12345/DeepLabV3-Plus/projects).

Model Architectures can be found [here](./models.md).

## Code to test Model

```python
from src.model import DeepLabV3Plus

model = DeepLabV3Plus()
model.summary()
```