###############################################################################
# なんか諸々便利であろう関数を作ったらここで呼び出す感じで ####################
###############################################################################

import csv
import os
import time
import yaml
import logging
from pathlib import Path
from attrdict import AttrDict
import urllib.request

import numpy as np
import pandas as pd
import torch
import torchvision

csv.field_size_limit(1000000000)


def load_config(config_path):
    """config(yaml)ファイルを読み込む

    Parameters
    ----------
    config_path : string
        config fileのパスを指定する

    Returns
    -------
    config : attrdict.AttrDict
        configを読み込んでattrdictにしたもの
    """
    with open(config_path, 'r', encoding='utf-8') as fi_:
        return AttrDict(yaml.load(fi_, Loader=yaml.SafeLoader))


def load_data(file_path):
    """csv or tsvを読み込む関数

    Parameters
    ----------
    config_path : string
        csv or tsv fileのパスを指定する

    Returns
    -------
    header : list
        ヘッダーのみを取り出したlist
    data : list
        対象のファイルのdata部分を取り出したlist
    """
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


def check_device():
    """pytorchが認識しているデバイスを返す関数

    Returns
    -------
    device : str
        cudaを使用する場合 `'cuda'` 、cpuで計算する場合は `'cpu'`
    """
    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    return DEVICE


def array_to_tensor(input_data, device=None):
    """np.array -> torch.tensor変換関数

    Parameters
    ----------
    input_data : np.array
        変換したいnp.array形式のデータ
    device : str, default None
        cudaを使用する場合 `'cuda'` 、cpuで計算する場合は `'cpu'`

        指定しない場合はtorchが認識している環境が選ばれるため、特に意思がなければデフォルトで良いはず。

    Returns
    -------
    output_data : torch.tensor
        input_dataをtensor型に変換したもの
    """
    if device is None:
        device = check_device()

    if input_data.dtype == float:
        # np.arrayはdoubleを前提として動いているが、
        # torchはdouble(float64)を前提としていない機能があるため、float32に変更する必要がある
        output_data = torch.tensor(input_data, dtype=torch.float32).contiguous().to(device)
    elif input_data.dtype == int:
        # 同様でtorchではlongが標準
        output_data = torch.tensor(input_data, dtype=torch.long).contiguous().to(device)
    return output_data


def tensor_to_array(input_data):
    """torch.tensor -> np.array変換関数

    Parameters
    ----------
    input_data : torch.tensor
        変換したいtorch.tensor形式のデータ

    Returns
    -------
    output_data : np.array
        input_dataをarray型に変換したもの
    """
    # numpyがメモリを共有するのを防ぐために以下の処理となる
    output_data = input_data.to('cpu').detach().numpy().copy()
    return output_data


def data_downloader():
    """the UC Irvine Machine Learning Repositoryからのdata downloader

    Examples
    --------
    # プロジェクトのホームディレクトリで
    $ python -c "from gp.utils.utils import data_downloader;data_downloader()"
    """
    index = {
        1: {'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00409/Daily_Demand_Forecasting_Orders.csv',
            'name': 'Daily Demand Forecasting Orders Data Set',
            'page': 'https://archive.ics.uci.edu/ml/datasets/Daily+Demand+Forecasting+Orders'},
        2: {'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00247/data_akbilgic.xlsx',
            'name': 'ISTANBUL STOCK EXCHANGE Data Set',
            'page': 'https://archive.ics.uci.edu/ml/datasets/ISTANBUL+STOCK+EXCHANGE'},
        3: {'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx',
            'name': 'Online Retail II Data Set',
            'page': 'https://archive.ics.uci.edu/ml/datasets/Online+Retail+II'},
        4: {'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip',
            'name': 'Beijing Multi-Site Air-Quality Data Data Set',
            'page': 'https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data'},
        5: {'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv',
            'name': 'Beijing PM2.5 Data Data Set',
            'page': 'https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data'},
        6: {'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00514/Bias_correction_ucl.csv',
            'name': 'Bias correction of numerical prediction model temperature forecast Data Set',
            'page': 'https://archive.ics.uci.edu/ml/datasets/Bias+correction+of+numerical+prediction+model+temperature+forecast'}
    }
    for k, v in index.items():
        print(f"number: {k}")
        print(f"name: {v['name']}")
        print(f"page: {v['page']}")
        print(f"url: {v['url']}")
        print()
    inp = int(input('欲しいデータの番号を入力して(1~6): '))
    if inp in {1, 2, 3, 4, 5, 6}:
        directory = './data'
        if not os.path.isdir(directory):
            os.makedirs(directory)
        urllib.request.urlretrieve(index[inp]['url'],
                                   os.path.join(directory, os.path.basename(index[inp]['url'])))
