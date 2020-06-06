###############################################################################
# ガウス過程で予想するメインとなる機能をここに ################################
###############################################################################

import numpy as np
import torch
import torchvision

class GaussianProcessRegressor(object):
    """学習・予測を行うメインとなる関数

    """
    def __init__(self, kernel='gauss', iv_method=True, mode='kissgp', num_iv=300, num_itr=30):
        """インスタンス変数
        インスタンス変数としてモジュールに渡すかfit時に渡すか要検討(num_itrなど)

        Parameters
        ----------
        kernel : string, optional (default=gauss)
            カーネル関数
        iv_method : bool, optional (default=True)
            補助変数法(inducing variable method)
        mode : string, optional (default='kissgp')
            補助変数法のmode
                SoD(subset of data approximation) : 部分データ法
                VB(variational bayesian methods) : 変分ベイズ法
                kissgp : KISS-GP法
        num_iv : int, optional (default=300)
            SoDの場合の補助変数の個数
        num_itr : int, optional (default=30)
            変分ベイズ方の時のイテレーションを回す回数
        silent : bool, optional (default=True)
            学習過程を出力を消すオプション
        """
        super(GaussianProcessRegressor, self).__init__()
        self.kernel = kernel
        self.iv_method = iv_method
        self.mode = mode
        self.num_iv = num_iv
        self.num_itr = num_itr

    def fit(self, x_train: np.array, y_train: np.array,
            silent=True):
        """学習用のメソッド

        Parameters
        ----------
        x_train : np.array
            説明変数のarray
        y_train : np.array
            目的変数のarray
        silent : bool, optional (default=True)
            学習過程を出力を消すオプション
        """
        self.cov = cov          # 必ず分散共分散行列は作るはず?
        self.model = model      # gpytorchなりの学習済みモデルを設定
        return self

    def train(self, x_train: np.array, y_train: np.array, **kwargs):
        """fitと同じ

        Parameters
        ----------
        x_train : np.array
            説明変数のarray
        y_train : np.array
            目的変数のarray
        **kwargs
            諸々オプション

        Returns
        -------
        out : 

        """
        self.fit(x_train, y_train, **kwargs)

    def predict(self, X: np.array) -> np.array:
        """予測用のメソッド

        Parameters
        ----------
        X : np.array
            説明変数のarray

        Returns
        -------
        y : np.array
            予測された目的変数のarray
                当然len(X) == len(y) -> Trueのはず
        """
        self.model              # 予測ではこれを使う
        return y

    def get_params(self) -> dict:
        """学習段階のパラメータを取り出す
        Returns
        -------
        params : dict
            パラメータの辞書
        """
        params = dict(
            kernel=self.kernel,
            iv_method=self.iv_method,
            mode=self.mode,
            num_iv=self.num_iv,
            num_itr=self.num_itr
        )
        return params

    def get_cov(self) -> np.array:
        """分散共分散行列を取り出す

        Returns
        -------
        cov : np.array
            分散共分散行列
        """
        return self.cov
