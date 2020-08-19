#!/usr/bin/env python

"""Module for training deeplabv3plus on camvid dataset."""

# !pylint:disable=wrong-import-position

import argparse
from argparse import RawTextHelpFormatter

print("[-] Importing tensorflow...")
import tensorflow as tf  # noqa: E402
print(f"[+] Done! Tensorflow version: {tf.version.VERSION}")

from deeplabv3plus.train import Trainer  # noqa: E402

from configs import CONFIG_MAP  # noqa: E402


if __name__ == "__main__":
    AVAILABLE_KEYS = "".join(map(lambda s: f"  {s}\n", CONFIG_MAP.keys()))

    PARSER = argparse.ArgumentParser(
        description=f"""
Runs DeeplabV3+ trainer with the given config setting.

Available config keys:
{AVAILABLE_KEYS}""",
        formatter_class=RawTextHelpFormatter
    )
    PARSER.add_argument('config_key', help="Key to use while looking up "
                        "configuration from the CONFIG_MAP dictionary.")
    PARSER.add_argument("--wandb_api_key",
                        help="Wandb API Key for logging run on Wandb",
                        default=None)
    ARGS = PARSER.parse_args()

    CONFIG = CONFIG_MAP[ARGS.config_key]
    if ARGS.wandb_api_key is not None:
        CONFIG['wandb_api_key'] = ARGS.wandb_api_key
        CONFIG['checkpoint_dir'] = "wandb://"

    TRAINER = Trainer(CONFIG_MAP[ARGS.CONFIG_KEY])
    HISTORY = TRAINER.train()
