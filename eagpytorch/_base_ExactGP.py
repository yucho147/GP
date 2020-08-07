#!/usr/bin/env python3

__all__ = [
    "ExactGPModel",
    "RunExactGP"
]

from random import seed

from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from numpy.random import seed as nseed
from numpy import (
    mean,
    ndarray
)
from torch import (
    manual_seed,
    no_grad
)
from torch.optim import (
    Adadelta,
    Adagrad,
    Adam,
    RMSprop,
    SGD
)

from ._const import (
    # likelihoods
    gaussianlikelihood,
    # mlls
    exactmarginalloglikelihood,
    # optimizers
    adadelta,
    adagrad,
    adam,
    rmsprop,
    sgd
)
from ._likelihoods import GaussianLikelihood
from .utils import (
    array_to_tensor,
    check_device,
    load_model,
    plot_kernel,
    save_model,
    set_kernel,
    _predict_obj,
    _sample_f
)


class ExactGPModel(ExactGP):
    """ExactGP用のモデル定義クラス

    Parameters
    ----------
    train_x : torch.tensor
        学習用データセットの説明変数
    train_y : torch.tensor
        学習用データセットの目的変数
    likelihood : :obj:`gpytorch.likelihoods`
        likelihoodのインスタンス
    ex_var_dim : int
        ARDカーネルのパラメータの個数。説明変数の数に対応する。

        `ex_var_dim=None` を指定するとRBFカーネルになる。
    kernel : str or :obj:`gpytorch.kernels`
        使用するカーネル関数を指定する

        基本はstrで指定されることを想定しているものの、自作のカーネル関数を入力することも可能
    **ker_conf : dict
        カーネル関数に渡す設定
    """
    def __init__(self,
                 train_x,
                 train_y,
                 likelihood,
                 ex_var_dim,
                 kernel,
                 **ker_conf):
        super(ExactGPModel, self).__init__(train_x,
                                           train_y,
                                           likelihood)
        self.mean_module = ConstantMean()
        _ker_conf = {'ard_num_dims': ex_var_dim}
        _ker_conf.update(ker_conf)
        self.covar_module = set_kernel(kernel, **_ker_conf)

    def forward(self, x):
        """ExactGPModelのforwardメソッド
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class RunExactGP(object):
    """ExactGPModelの実行クラス

    Parameters
    ----------
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
        likelihoodとして使用するクラス名が指定される。

        - :obj:`'GaussianLikelihood'`, :obj:`'GL'` : likelihoodにガウス分布を仮定したガウス過程を行う場合

    optimizer : str, default :obj:`'Adam'`
        optimizerとして使用するクラス名が指定される。下記から選択する。

        - :obj:`'Adam'`
        - :obj:`'sgd'`
        - :obj:`'RMSprop'`
        - :obj:`'Adadelta'`
        - :obj:`'Adagrad'`

    mll : str, default :obj:`'ExactMarginalLogLikelihood'`
        確率分布の周辺化の方法のクラス名が指定される。

        - :obj:`'ExactMarginalLogLikelihood'`

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
    def __init__(self,
                 *,
                 kernel='RBFKernel',
                 likelihood='GaussianLikelihood',
                 optimizer='Adam',
                 mll='ExactMarginalLogLikelihood',
                 ard_option=True,
                 ker_conf=dict(),
                 opt_conf=dict(),
                 mll_conf=dict(),
                 random_state=None):
        if isinstance(random_state, int):
            seed(random_state)
            nseed(random_state)
            manual_seed(random_state)
        self.device = check_device()
        self._kernel = kernel
        self._likelihood = likelihood
        self._optimizer = optimizer
        self._mll = mll
        self.ard_option = ard_option
        self.epoch = 0
        self.model = None  # 空のmodelを作成しないとloadできない
        self.mll = None    # 空のmodelを作成しないとloadできない
        self.optimizer = None  # 空のmodelを作成しないとloadできない
        self._ker_conf = ker_conf
        self._opt_conf = opt_conf
        self._mll_conf = mll_conf
        self.loss = []

    def _set_likelihood(self):
        """likelihoodとしてself._likelihoodの指示の元、インスタンスを立てるメソッド
        """
        if self._likelihood in gaussianlikelihood:
            return GaussianLikelihood().to(self.device)
        else:
            raise ValueError(f'likelihood={self._likelihood}は用意されていません')

    def _set_mll(self, mll_conf):
        """mllとしてself._mllの指示の元、インスタンスを立てるメソッド
        """
        # mllのインスタンスを立てる
        if self._mll in exactmarginalloglikelihood:
            return ExactMarginalLogLikelihood(
                self.likelihood,
                self.model
            )
        else:
            raise ValueError(f'mll={self._mll}は用意されていません')

    def _set_optimizer(self, lr, opt_conf):
        """optimizerとしてself._optimizerの指示の元、インスタンスを立てるメソッド
        """
        if self._optimizer in adam:
            return Adam([
                {'params': self.model.parameters()}
            ], lr=lr, **opt_conf)
        elif self._optimizer in sgd:
            return SGD([
                {'params': self.model.parameters()}
            ], lr=lr, **opt_conf)
        elif self._optimizer in rmsprop:
            return RMSprop([
                {'params': self.model.parameters()}
            ], lr=lr, **opt_conf)
        elif self._optimizer in adadelta:
            return Adadelta([
                {'params': self.model.parameters()}
            ], lr=lr, **opt_conf)
        elif self._optimizer in adagrad:
            return Adagrad([
                {'params': self.model.parameters()}
            ], lr=lr, **opt_conf)
        else:
            raise ValueError(f'optimizer={self._optimizer}は用意されていません')

    def set_model(self,
                  train_x,
                  train_y,
                  *,
                  lr=1e-3):
        """使用するモデルのインスタンスを立てるメソッド

        Parameters
        ----------
        train_x : np.array or torch.tensor
            学習用データセットの説明変数
        train_y : np.array or torch.tensor
            学習用データセットの目的変数
        lr : float
            学習率
        """
        if type(train_x) == ndarray:
            train_x = (array_to_tensor(train_x),)
            train_x = tuple(tri.unsqueeze(-1) if tri.ndimension() == 1 else tri for tri in train_x)[0]
        if type(train_y) == ndarray:
            train_y = array_to_tensor(train_y)
        ard_option = self.ard_option
        ex_var_dim = train_x.shape[1]

        kernel = self._kernel
        ker_conf = self._ker_conf

        # likelihoodのインスタンスを立てる
        self.likelihood = self._set_likelihood()

        # ここで上記モデルのインスタンスを立てる
        if ard_option:
            self.model = ExactGPModel(
                train_x,
                train_y,
                self.likelihood,
                ex_var_dim=ex_var_dim,
                kernel=kernel,
                **ker_conf
            ).to(self.device)
        else:
            self.model = ExactGPModel(
                train_x,
                train_y,
                self.likelihood,
                ex_var_dim=None,
                kernel=kernel,
                **ker_conf
            ).to(self.device)

        # mllのインスタンスを立てる
        self.mll = self._set_mll(self._mll_conf)
        # optimizerのインスタンスを立てる
        self.optimizer = self._set_optimizer(lr, opt_conf=self._opt_conf)

    def fit(self, epochs, *, test_x=None, test_y=None, verbose=True):
        """学習用メソッド

        Parameters
        ----------
        epochs : int
            エポック数
        test_x : np.ndarray or torch.tensor, default None
            説明変数のテストデータ
        test_y : np.ndarray or torch.tensor, default None
            目的変数のテストデータ

            もし test_x, test_y を設定している場合エポックごとにテストデータに対するlossも表示されるように設定される
        verbose : bool, default True
            表示形式
        """
        if type(test_x) == ndarray:
            test_x = (array_to_tensor(test_x),)
            test_x = tuple(tri.unsqueeze(-1) if tri.ndimension() == 1 else tri for tri in test_x)[0]
        if type(test_y) == ndarray:
            test_y = array_to_tensor(test_y)

        for epoch in range(epochs):
            train_loss = []
            test_loss = []
            self.model.train()
            self.likelihood.train()
            self.optimizer.zero_grad()
            output = self.model(self.model.train_inputs[0])
            loss = - self.mll(output, self.model.train_targets)
            loss.backward()
            self.optimizer.step()

            self.loss.append(loss.item())
            train_loss.append(loss.item())

            self.model.eval()
            self.likelihood.eval()
            if test_x is not None and test_y is not None:
                with no_grad():
                    output = self.model(test_x)
                    loss = - self.mll(output, test_y)
                    test_loss.append(loss.item())

            is_display_timing = epoch % (epochs // 10) == 0 if epochs >= 10 else True
            if is_display_timing or epoch == epochs - 1 and verbose:
                if test_loss:
                    print(f'Epoch {epoch + 1}/{epochs}'
                          + f' - Train Loss: {mean(train_loss):.5f} /'
                          + f' Test Loss: {mean(test_loss):.5f}')
                else:
                    print(f'Epoch {epoch + 1}/{epochs}'
                          + f' - Train Loss: {mean(train_loss):.5f}')
        # TODO: 追加学習のために再学習の際、self.epochを利用する形にする
        self.epoch = epoch + 1

    def predict(self, X, *, cl=0.6827, sample_num=None, sample_f_num=None):
        """予測用メソッド

        Parameters
        ----------
        X : np.array or torch.tensor
            入力説明変数
        cl : float default 0.6827(1sigma)
            信頼区間
        sample_num : int default None
            yのサンプル数
        sample_f_num : int default None
            fのサンプル数

        Returns
        -------
        output : object
            予測された目的変数のオブジェクト。下記の属性が用意されている。

            - output.mean : 予測された目的変数の平均値
            - output.upper : 予測された目的変数の信頼区間の上限
            - output.lower : 予測された目的変数の信頼区間の下限
            - output.samples : 入力説明変数に対する予測値yのサンプル(sample_num個サンプルされる)
            - output.samples_f : 入力説明変数に対する予測関数fのサンプル(sample_f_num個サンプルされる)
        """
        if type(X) == ndarray:
            X = array_to_tensor(X)
        if not 0.5 < cl < 1.0:
            raise ValueError(f'cl={cl}が正しい値ではありません')
        self.model.eval()
        self.likelihood.eval()
        with no_grad():
            predicts = self.likelihood(self.model(X))
            if self._likelihood in {'GaussianLikelihood', 'GL'}:
                predicts_f = self.model(X)
            else:
                predicts_f = None
        output = _predict_obj(predicts, cl, sample_num)
        output.samples_f = _sample_f(predicts_f, sample_f_num)
        return output

    def save(self, file_path):
        """モデルのsaveメソッド

        Parameters
        ----------
        file_path : str
            モデルの保存先のパスとファイル名
        """
        data = dict(
            epoch=self.epoch,
            model=self.model,
            likelihood=self.likelihood,
            mll=self.mll,
            optimizer=self.optimizer,
            loss=self.loss
        )
        save_model(file_path, **data)

    def load(self, file_path):
        """モデルのloadメソッド

        Parameters
        ----------
        file_path : str
            モデルの保存先のパスとファイル名
        """
        data = dict(
            epoch=self.epoch,
            model=self.model,
            likelihood=self.likelihood,
            mll=self.mll,
            optimizer=self.optimizer,
            loss=self.loss
        )
        (self.epoch,
         self.model,
         self.likelihood,
         self.mll,
         self.optimizer,
         self.loss) = load_model(file_path, **data)

    def kernel_coeff(self):
        """kernelの係数を出力するメソッド

        Returns
        -------
        output_dict : dict
            カーネル関数の係数

            `ard_option=True` の場合、 $\Theta$ が各々の説明変数ごとに重みを変えて更新され、出力される

        Warning
        --------
        RBFKernelの場合、各説明変数の重要度 $\eta$ は出力される `'base_kernel.raw_lengthscale'`
        の逆数の2乗に対応する
        """
        # TODO: kernel関数をスイッチさせ、それに応じてわかりやすい形に変形する
        output_dict = self.model.covar_module.state_dict()
        return output_dict

    def plot_kernel(self, *, kernel=None, plot_range=None, **kwargs):
        """カーネル関数のプロット

        Parameters
        ----------
        kernel : str or :obj:`gpytorch.kernels`, default None
            使用するカーネル関数を指定する

        plot_range : tuple, default None
            プロットする幅

        **kwargs : dict
            カーネル関数に渡す設定
        """
        if kernel is None:
            if kwargs:
                temp_kernel = set_kernel(self._kernel, **self._ker_conf)
            else:
                temp_kernel = set_kernel(self._kernel, **kwargs)
        else:
            if kwargs:
                temp_kernel = set_kernel(kernel, **self._ker_conf)
            else:
                temp_kernel = set_kernel(kernel, **kwargs)

        plot_kernel(temp_kernel, plot_range, **kwargs)
