#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""__init__ module for configs. Register your config file here by adding it's
entry in the CONFIG_MAP as shown.
"""

import config.camvid_resnet50
import config.camvid_resnet101
import config.camvid_mobilenetv2
import config.human_parsing_resnet50
import config.human_parsing_resnet101
import config.human_parsing_mobilenetv2


CONFIG_MAP = {
    'camvid_resnet50': config.camvid_resnet50.CONFIG,
    'camvid_resnet101': config.camvid_resnet101.CONFIG,
    'camvid_mobilenetv2': config.camvid_mobilenetv2.CONFIG,
    'human_parsing_resnet50': config.human_parsing_resnet50.CONFIG,
    'human_parsing_resnet101': config.human_parsing_resnet101.CONFIG,
    'human_parsing_mobilenetv2': config.human_parsing_mobilenetv2.CONFIG,
}
