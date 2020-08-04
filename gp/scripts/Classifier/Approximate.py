from ..base_ApproximateGP import RunApproximateGP
from ..const import regs


class Approximate(RunApproximateGP):
    """ApproximateGPでの分類クラス

    ApproximateGPModelをラップし、学習・予測・プロット等を司る

    Parameters
    ----------
    inducing_points_num : int or float
        補助変数の個数(int)

        もし 0 < inducing_points_num < 1 が渡された場合学習用データの len と inducing_points_num の積が補助変数の個数として設定される
    kernel : str or :obj:`gpytorch.kernels`, default 'RBFKernel
        使用するカーネル関数を指定する

        基本はstrで指定されることを想定しているものの、自作のカーネル関数を入力することも可能
    likelihood : str, default 'BernoulliLikelihood'
        likelihoodとして使用するクラス名が指定される
    optimizer : str, default 'Adam'
        optimizerとして使用するクラス名が指定される
    mll : str, default 'VariationalELBO'
        確率分布の周辺化の方法のクラス名が指定される
    ard_option : bool, default True
        ARDカーネルを利用するかが指定される

        もし :obj:`RunApproximateGP.kernel_coeff` を利用する場合 `ard_option=True` を選択する
    ker_conf : dict, default dict()
        カーネル関数に渡す設定一覧辞書
    mll_conf : dict, default dict()
        mllに渡す設定一覧辞書
    opt_conf : dict, default dict()
        optimizerに渡す設定一覧辞書
    random_state : int, default None
        seedの固定
    """
    def __init__(self, *args, **kwargs):
        if kwargs.get('likelihood') is None:
            kwargs['likelihood'] = 'BernoulliLikelihood'
        if kwargs.get('likelihood') in regs:
            raise ValueError(f"Classifierに{kwargs.get('likelihood')}は使えない")
        else:
            super(Approximate, self).__init__(*args, **kwargs)
