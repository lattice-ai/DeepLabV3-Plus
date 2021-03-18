from typing import List
import tensorflow as tf
from typing import Tuple
from matplotlib import pyplot as plt


def bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value])
    )


def split_list(given_list: List, chunk_size: int):
    return [
        given_list[offs: offs + chunk_size]
        for offs in range(0, len(given_list), chunk_size)
    ]


def create_example(image_path, label_path):
    return {
        'image': bytes_feature(tf.io.read_file(image_path).numpy()),
        'label': bytes_feature(tf.io.read_file(label_path).numpy())
    }


def plot_result(
    images, captions: List[str], title: str, figsize: Tuple[int, int]):
    fig = plt.figure(figsize=figsize)
    plt.suptitle(
        'Label: ' + title[0], fontsize=20, fontweight='bold')
    for index in range(len(captions)):
        fig.add_subplot(
            1, len(captions), index + 1
        ).set_title(captions[index])
        _ = plt.imshow(images[index])
        plt.axis(False)
    plt.show()
