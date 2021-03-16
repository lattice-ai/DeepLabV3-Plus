from typing import List
import tensorflow as tf


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
        'image': bytes_feature(tf.io.read_file(image_path)),
        'label': bytes_feature(tf.io.read_file(label_path))
    }
