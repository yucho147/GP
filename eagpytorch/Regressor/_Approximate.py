#!/usr/bin/env python3

__all__ = [
    "Approximate"
]

from .. import RunApproximateGP
from .._const import clas


class Approximate(RunApproximateGP):
    """ApproximateGPModelの実行クラス

    Parameters
    ----------
    inducing_points_num : int or float
        補助変数の個数(int)

        もし 0 < inducing_points_num < 1 が渡された場合、学習用データの len と
        inducing_points_num の積が補助変数の個数として設定される
    kernel : str or :obj:`gpytorch.kernels`, default :obj:`'RBFKernel'`
        使用するカーネル関数を指定する。下記から選択する。

        - :obj:`'CosineKernel'`
        - :obj:`'LinearKernel'`
        - :obj:`'MaternKernel'`
        - :obj:`'PeriodicKernel'`
        - :obj:`'RBFKernel'`
        - :obj:`'RQKernel'`
        - :obj:`'SpectralMixtureKernel'`

        基本はstrで指定されることを想定しているものの、 :obj:`gpytorch.kernels`
        を用いた自作のカーネル関数を入力することも可能
    likelihood : str, default :obj:`'GaussianLikelihood'`
        likelihoodとして使用するクラス名が指定される。下記から選択する。

        - :obj:`'GaussianLikelihood'`, :obj:`'GL'` : likelihoodにガウス分布を仮定したガウス過程を行う場合
        - :obj:`'PoissonLikelihood'`, :obj:`'PL'` : likelihoodにポアソン分布を仮定したポアソン回帰ガウス過程モデルを行う場合

    optimizer : str, default :obj:`'Adam'`
        optimizerとして使用するクラス名が指定される。下記から選択する。

        - :obj:`'Adam'`
        - :obj:`'sgd'`
        - :obj:`'RMSprop'`
        - :obj:`'Adadelta'`
        - :obj:`'Adagrad'`

    mll : str, default :obj:`'VariationalELBO'`
        確率分布の周辺化の方法のクラス名が指定される。下記から選択する。

        - :obj:`'VariationalELBO'`, :obj:`'VELBO'`
        - :obj:`'PredictiveLogLikelihood'`, :obj:`'PLL'`
        - :obj:`'GammaRobustVariationalELBO'`, :obj:`'GRVELBO'`

    ard_option : bool, default True
        ARDカーネルを利用するかが指定される

        もし :obj:`kernel_coeff` を利用する場合 `ard_option=True` を選択する
    ker_conf : dict, default dict()
        カーネル関数に渡す設定
    mll_conf : dict, default dict()
        mllに渡す設定
    opt_conf : dict, default dict()
        optimizerに渡す設定
    random_state : int, default None
        seedの固定
    """
    def __init__(self, *args, **kwargs):
        if kwargs.get('likelihood') in clas:
            raise ValueError(f"Regressorに{kwargs.get('likelihood')}は使えない")
        else:
            super(Approximate, self).__init__(*args, **kwargs)

    def set_model(self, *args, **kwargs):
        super(Approximate, self).set_model(*args, **kwargs)

    def fit(self, *args, **kwargs):
        super(Approximate, self).fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """予測用メソッド

        Parameters
        ----------
        X : np.array or torch.tensor
            入力説明変数
        cl : float default 0.6827(1sigma)
            信頼区間
        sample_num : int default None
            yのサンプル数
        sample_f_num : int default None
            fのサンプル数

        Returns
        -------
        output : object
            予測された目的変数のオブジェクト。下記の属性が用意されている。

            - output.mean : 予測された目的変数の平均値
            - output.upper : 予測された目的変数の信頼区間の上限
            - output.lower : 予測された目的変数の信頼区間の下限
            - output.samples : 入力説明変数に対する予測値yのサンプル(sample_num個サンプルされる)
            - output.samples_f : 入力説明変数に対する予測関数fのサンプル(sample_f_num個サンプルされる)
        """
        return super(Approximate, self).predict(*args, **kwargs)

    def save(self, *args, **kwargs):
        super(Approximate, self).save(*args, **kwargs)

    def load(self, *args, **kwargs):
        super(Approximate, self).load(*args, **kwargs)

    def kernel_coeff(self, *args, **kwargs):
        return super(Approximate, self).kernel_coeff(*args, **kwargs)

    def plot_kernel(self, *args, **kwargs):
        super(Approximate, self).plot_kernel(*args, **kwargs)
