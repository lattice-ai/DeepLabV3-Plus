import os
from tqdm import tqdm
from typing import List
import tensorflow as tf
from tqdm.notebook import tqdm as tqdm_notebook

from .commons import create_example, split_list


class TFRecordCreator:

    def __init__(
            self, output_directory: str, dataset_name: str,
            shard_size: int = 128, is_notebook: bool = False):
        self.output_directory = output_directory
        self.dataset_name = dataset_name
        self.shard_size = shard_size
        self.is_notebook = is_notebook

    def _create_tfrecord_files(
            self, image_shards: List[List[str]], label_shards: List[List[str]], dataset_split: str):
        write_directory = os.path.join(
            self.output_directory,
            '{}-{}'.format(self.dataset_name, dataset_split)
        )
        if not os.path.exists(write_directory):
            os.mkdir(write_directory)
        progress_bar = tqdm_notebook if self.is_notebook else tqdm
        for shard, image_shard in enumerate(progress_bar(image_shards)):
            shard_size = len(image_shard)
            label_shard = label_shards[shard]
            file_path = os.path.join(
                write_directory, '{}-{}-{}.tfrec'.format(
                    dataset_split, shard, shard_size
                )
            )
            with tf.io.TFRecordWriter(file_path) as out_files:
                for data_index, image_data in enumerate(image_shard):
                    example = create_example(
                        image_path=image_data, label_path=label_shard[data_index])
                    out_files.write(example.SerializeToString())

    def create(self, image_files: List[str], label_files: List[str], dataset_split: str):
        image_shards = split_list(image_files, chunk_size=self.shard_size)
        label_shards = split_list(label_files, chunk_size=self.shard_size)
        print('[+] Creating {} TFRecords...'.format(dataset_split))
        self._create_tfrecord_files(
            image_shards=image_shards, label_shards=label_shards,
            dataset_split=dataset_split
        )
