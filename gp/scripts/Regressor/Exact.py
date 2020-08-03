from ..base_ExactGP import RunExactGP
from ..const import clas


class Exact(RunExactGP):
    """ExactGPでの回帰クラス
    ExactGPModelをラップし、学習・予測・プロット等を司る

    Parameters
    ----------
    kernel : str or :obj:`gpytorch.kernels`, default 'RBFKernel
        使用するカーネル関数を指定する
    likelihood : str
        likelihoodとして使用するクラス名が指定される
    optimizer : str
        optimizerとして使用するクラス名が指定される
    mll : str
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
        if kwargs.get('likelihood') in clas:
            raise ValueError(f"regressionに{kwargs.get('likelihood')}は使えない")
        else:
            super(Exact, self).__init__(*args, **kwargs)
