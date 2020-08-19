#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""__init__ module for configs. Register your config file here by adding it's
entry in the CONFIG_MAP as shown.
"""

import configs.camvid_resnet50
import configs.human_parsing_resnet50


CONFIG_MAP = {
    'camvid_resnet50': configs.camvid_resnet50.CONFIG,
    'human_parsing_resnet50': configs.human_parsing_resnet50
}
