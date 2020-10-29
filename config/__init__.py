#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""__init__ module for configs. Register your config file here by adding it's
entry in the CONFIG_MAP as shown.
"""

import config.camvid_resnet50
import config.human_parsing_resnet50


CONFIG_MAP = {
    'camvid_resnet50': config.camvid_resnet50.CONFIG,
    'human_parsing_resnet50': config.human_parsing_resnet50.CONFIG
}
