#!/usr/bin/env python3

__all__ = [
    "array_to_tensor",
    "check_device",
    "load_config",
    "load_data",
    "plot_kernel",
    "set_kernel",
    "tensor_to_array"
]

from attrdict import AttrDict
from os import makedirs
from os.path import (
    basename,
    isdir,
    join,
    splitext
)
from urllib.request import urlretrieve
import csv
import yaml

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import (
    CosineKernel,
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    RBFKernel,
    RQKernel,
    ScaleKernel,
    SpectralMixtureKernel
)
from gpytorch import kernels
from pyro.distributions import Poisson, Bernoulli
import matplotlib.pyplot as plt
import numpy as np
import torch


csv.field_size_limit(1000000000)


def load_config(config_path):
    """config(yaml)ファイルを読み込む

    Parameters
    ----------
    config_path : string
        config fileのパスを指定する

    Returns
    -------
    config : attrdict.AttrDict
        configを読み込んでattrdictにしたもの
    """
    with open(config_path, 'r', encoding='utf-8') as fi_:
        return AttrDict(yaml.load(fi_, Loader=yaml.SafeLoader))


def load_data(file_path):
    """csv or tsvを読み込む関数

    Parameters
    ----------
    config_path : string
        csv or tsv fileのパスを指定する

    Returns
    -------
    header : list
        ヘッダーのみを取り出したlist
    data : list
        対象のファイルのdata部分を取り出したlist
    """
    header = []
    data = []
    with open(file_path, 'r', newline="\n") as fi_:
        root, ext = splitext(file_path)
        if ext == '.tsv':
            reader = csv.reader(fi_, delimiter='\t')
        elif ext == '.csv':
            reader = csv.reader(fi_, delimiter=',')
        header = next(reader)
        for d in reader:
            data.append([float(i) for i in d])
    return header, data


def check_device():
    """pytorchが認識しているデバイスを返す関数

    Returns
    -------
    device : str
        cudaを使用する場合 `'cuda'` 、cpuで計算する場合は `'cpu'`
    """
    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    return DEVICE


def array_to_tensor(input_data, device=None):
    """np.array -> torch.tensor変換関数

    Parameters
    ----------
    input_data : np.array
        変換したいnp.array形式のデータ
    device : str, default None
        cudaを使用する場合 `'cuda'` 、cpuで計算する場合は `'cpu'`

        指定しない場合はtorchが認識している環境が選ばれるため、特に意思がなければデフォルトで良いはず。

    Returns
    -------
    output_data : torch.tensor
        input_dataをtensor型に変換したもの
    """
    if device is None:
        device = check_device()

    if input_data.dtype == float:
        # np.arrayはdoubleを前提として動いているが、
        # torchはdouble(float64)を前提としていない機能があるため、float32に変更する必要がある
        output_data = torch.tensor(input_data,
                                   dtype=torch.float32).contiguous().to(device)
    elif input_data.dtype == int:
        # 同様でtorchではlongが標準
        output_data = torch.tensor(input_data,
                                   dtype=torch.long).contiguous().to(device)
    return output_data


def tensor_to_array(input_data):
    """torch.tensor -> np.array変換関数

    Parameters
    ----------
    input_data : torch.tensor
        変換したいtorch.tensor形式のデータ

    Returns
    -------
    output_data : np.array
        input_dataをarray型に変換したもの
    """
    # numpyがメモリを共有するのを防ぐために以下の処理となる
    output_data = input_data.to('cpu').detach().numpy().copy()
    return output_data


def data_downloader():
    """the UC Irvine Machine Learning Repositoryからのdata downloader

    Examples
    --------
    プロジェクトのホームディレクトリから::

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
    inp = int(input('欲しいデータの番号を入力して(1~6): '))
    if inp in index.keys():
        directory = './data'
        if not isdir(directory):
            makedirs(directory)
        urlretrieve(index[inp]['url'],
                    join(directory, basename(index[inp]['url'])))


def save_model(file_path, *, epoch, model, likelihood, mll, optimizer, loss):
    """モデルの保存関数

    Parameters
    ----------
    file_path : str
        モデルの保存先のパスとファイル名
    epoch : int
        現在のエポック数
    model : :obj:`gpytorch.models`
        学習済みのモデルのオブジェクト
    likelihood : :obj:`gpytorch.likelihoods`
        学習済みのlikelihoodsのオブジェクト
    mll : :obj:`gpytorch.mlls`
        学習済みのmllsのオブジェクト
    optimizer : :obj:`torch.optim`
        学習済みのoptimのオブジェクト
    loss : list
        現在のエポックまでの経過loss
    """
    torch.save({'epoch': epoch,
                'model': model.state_dict(),
                'likelihood': likelihood.state_dict(),
                'mll': mll.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss},
               file_path)


def load_model(file_path, *, epoch, model, likelihood, mll, optimizer, loss):
    """モデルの保存関数

    Parameters
    ----------
    file_path : str
        モデルの保存先のパスとファイル名
    epoch : int
        現在のエポック数
    model : :obj:`gpytorch.models`
        学習済みのモデルのオブジェクト
    likelihood : :obj:`gpytorch.likelihoods`
        学習済みのlikelihoodsのオブジェクト
    mll : :obj:`gpytorch.mlls`
        学習済みのmllsのオブジェクト
    optimizer : :obj:`torch.optim`
        学習済みのoptimのオブジェクト
    loss : list
        現在のエポックまでの経過loss

    Returns
    -------
    epoch : int
        現在のエポック数
    model : :obj:`gpytorch.models`
        学習済みのモデルのオブジェクト
    likelihood : :obj:`gpytorch.likelihoods`
        学習済みのlikelihoodsのオブジェクト
    mll : :obj:`gpytorch.mlls`
        学習済みのmllsのオブジェクト
    optimizer : :obj:`torch.optim`
        学習済みのoptimのオブジェクト
    loss : list
        現在のエポックまでの経過loss
    """
    temp = torch.load(file_path)
    epoch = temp['epoch']
    model.load_state_dict(temp['model'])
    likelihood.load_state_dict(temp['likelihood'])
    mll.load_state_dict(temp['mll'])
    optimizer.load_state_dict(temp['optimizer'])
    loss = temp['loss']

    return epoch, model, likelihood, mll, optimizer, loss


def set_kernel(kernel, **kwargs):
    """kernelsを指定する

    Parameters
    ----------
    kernel : str or :obj:`gpytorch.kernels`
        使用するカーネル関数を指定する

        基本はstrで指定されることを想定しているものの、自作のカーネル関数を入力することも可能

    **kwargs : dict
        カーネル関数に渡す設定

    Returns
    -------
    out : :obj:`gpytorch.kernels`
        カーネル関数のインスタンス
    """
    if isinstance(kernel, str):
        if kernel in {'CosineKernel'}:
            return ScaleKernel(
                CosineKernel(**kwargs)
            )
        elif kernel in {'LinearKernel'}:
            return ScaleKernel(
                LinearKernel(**kwargs)
                )
        elif kernel in {'MaternKernel'}:
            return ScaleKernel(
                MaternKernel(**kwargs)
                )
        elif kernel in {'PeriodicKernel'}:
            return ScaleKernel(
                PeriodicKernel(**kwargs)
                )
        elif kernel in {'RBFKernel'}:
            return ScaleKernel(
                RBFKernel(**kwargs)
            )
        elif kernel in {'RQKernel'}:
            return ScaleKernel(
                RQKernel(**kwargs)
            )
        elif kernel in {'SpectralMixtureKernel'}:
            # SpectralMixtureKernelはScaleKernelを使えない
            return SpectralMixtureKernel(**kwargs)
        else:
            raise ValueError
    elif kernels.__name__ in str(type(kernel)):
        return kernel


def plot_kernel(kernel, plot_range=None, **kwargs):
    """カーネルの概形をプロットする関数

    Parameters
    ----------
    kernel : str or :obj:`gpytorch.kernels`
        使用するカーネル関数を指定する

    plot_range : tuple, default None
        プロットする幅

    **kwargs : dict
        カーネル関数に渡す設定
    """
    if plot_range is None:
        plot_range = torch.linspace(-1.5, 1.5)
    elif isinstance(plot_range, tuple):
        if len(plot_range) == 2:
            plot_range = torch.linspace(plot_range[0], plot_range[1])
        else:
            ValueError
    else:
        ValueError

    kernel = set_kernel(kernel, **kwargs)
    plt.plot(kernel(plot_range).numpy()[50])
    plt.xticks(
        [i for i in np.linspace(0, 100, 5)],
        [f'{i:.2f}' for i in np.linspace(
            plot_range.min().item(),
            plot_range.max().item(),
            5
        )]
    )
    plt.xlabel('x')
    plt.ylabel(
        f'kernel ({((plot_range.max()+plot_range.min())/2).item():.2f},  x)'
    )
    plt.show()


def _predict_obj(input, cl=0.6827, sample_num=None):
    """predictメソッドで利用するオブジェクトを返す関数

    Parameters
    ----------
    input : object
        likelihoodsの返り値
    cl : float default 0.6827(1sigma)
        信頼区間[%]
    sample_num : int default None
        サンプル数

    Returns
    -------
    output : object
        予測された目的変数のオブジェクト

        - output.mean : 予測された目的変数の平均値
        - output.upper : 予測された目的変数の信頼区間の上限
        - output.lower : 予測された目的変数の信頼区間の下限
        - output.samples : 入力説明変数に対する予測サンプル(sample_num個サンプルされる)
        - output.probs : BernoulliLikelihood を指定した際に、2値分類の予測確率
                         このとき mean,upper,lower は output に追加されない
    """
    class OutPut:
        pass
    output = OutPut

    if isinstance(input, MultivariateNormal):
        std = input.stddev
        mean = input.mean
        output.mean = tensor_to_array(mean)
        dist = torch.distributions.Normal(mean, std)

        output.upper = tensor_to_array(dist.icdf(torch.tensor([(1.+cl)/2.])))
        output.lower = tensor_to_array(dist.icdf(torch.tensor([(1.-cl)/2.])))
        if sample_num:
            output.samples = tensor_to_array(
                input.sample(torch.Size([sample_num]))
            )
        else:
            output.samples = None
    elif isinstance(input, Poisson):
        output.mean = tensor_to_array(input.mean)
        percentiles = [(1.-cl)/2., (1.+cl)/2.]
        output.lower, output.upper = _percentiles_from_samples(
            input.sample(torch.Size([1000])),
            percentiles
        )
        output.lower = tensor_to_array(output.lower)
        output.upper = tensor_to_array(output.upper)
        if sample_num:
            output.samples = tensor_to_array(
                input.sample(torch.Size([sample_num]))
            )
        else:
            output.samples = None
    elif isinstance(input, Bernoulli):
        output.probs = tensor_to_array(input.probs)
        if sample_num:
            output.samples = tensor_to_array(
                input.sample(torch.Size([sample_num]))
            )
        else:
            output.samples = None
    else:
        percentiles = [(1.-cl)/2., 0.5, (1.+cl)/2.]
        output.lower, output.mean, output.upper = _percentiles_from_samples(
            input.sample(torch.Size([1000])),
            percentiles
        )
        output.lower = tensor_to_array(output.lower)
        output.mean = tensor_to_array(output.mean)
        output.upper = tensor_to_array(output.upper)
        if sample_num:
            output.samples = tensor_to_array(
                input.sample(torch.Size([sample_num]))
            )
        else:
            output.samples = None

    return output


def _percentiles_from_samples(samples, percentiles=[0.05, 0.5, 0.95]):
    """サンプルされたデータセットからパーセンタイル点を求め、関数形をスムージングする関数
    Parameters
    ----------
    samples : object
        likelihoodsの返り値
    percentiles : list default [0.05, 0.5, 0.95]
        知りたいパーセンタイル点

    Returns
    -------
    percentiles_from_samples : tensor
        パーセンタイル点の値
    """
    num_samples = samples.size(0)
    samples = samples.sort(dim=0)[0]

    # Get samples corresponding to percentile
    percentile_samples = [
        samples[int(num_samples * percentile)]
        for percentile in percentiles
    ]

    # Smooth the samples
    kernel = torch.full((1, 1, 5), fill_value=0.2)
    percentiles_samples = [
        torch.nn.functional.conv1d(
            percentile_sample.view(1, 1, -1),
            kernel,
            padding=2
        ).view(-1)
        for percentile_sample in percentile_samples
    ]

    return percentiles_samples


def _sample_f(predicts_f, sample_f_num):
    """関数fのサンプル

    Parameters
    ----------
    predicts_f : :obj:`MultivariateNormal` or None
        事前分布fの予測モデル出力
    sample_f_num : int or None
        サンプル数

    Returns
    -------
    out : numpy.array or None
        サンプルされた関数形
    """
    if predicts_f is None:
        return None
    else:
        if sample_f_num is not None:
            return tensor_to_array(predicts_f.sample(torch.Size([sample_f_num])))
        else:
            return None
