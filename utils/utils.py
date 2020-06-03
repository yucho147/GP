###############################################################################
# なんか諸々便利であろう関数を作ったらここで呼び出す感じで ####################
###############################################################################

import os
import time
import yaml
import logging
from pathlib import Path
from attrdict import AttrDict

import numpy as np
import pandas as pd
import torch
import torchvision

csv.field_size_limit(1000000000)

def load_config(config_path):
    """config(yaml)ファイルを読み込む
    一旦、yamlを想定する
    """
    with open(config_path, 'r', encoding='utf-8') as fi_:
        return AttrDict(yaml.load(fi_, Loader=yaml.SafeLoader))


def load_data(file_path):
    header = []
    data = []
    with open(file_path, 'r', newline="\n") as fi_:
        root, ext = os.path.splitext(file_path)
        if ext == '.tsv':
            reader = csv.reader(fi_, delimiter='\t')
        elif ext == '.csv':
            reader = csv.reader(fi_, delimiter=',')
        header = next(reader)
        for d in reader:
            data.append(d)
    return header, data
