#!/usr/bin/env python

"""Module for training deeplabv3plus on camvid dataset."""
import argparse
from argparse import RawTextHelpFormatter

from deeplabv3plus.train import Trainer

from configs import CONFIG_MAP


if __name__ == "__main__":
    AVAILABLE_KEYS = "".join(map(lambda s: f"  {s}\n", CONFIG_MAP.keys()))

    PARSER = argparse.ArgumentParser(
        description=\
f"""Runs DeeplabV3+ trainer with the given config setting.

Available config keys:
{AVAILABLE_KEYS}""",
        formatter_class=RawTextHelpFormatter
    )
    PARSER.add_argument('CONFIG_KEY', help="Key to use while looking up "
                        "configuration from the CONFIG_MAP dictionary.")
    ARGS = PARSER.parse_args()

    TRAINER = Trainer(CONFIG_MAP[ARGS.CONFIG_KEY])
    HISTORY = TRAINER.train()
