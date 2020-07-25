import random

from gpytorch.models import ExactGP
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from gp.utils.utils import (array_to_tensor,
                            check_device,
                            load_model,
                            plot_kernel,
                            save_model,
                            set_kernel,
                            tensor_to_array)

from .likelihoods import GaussianLikelihood


class ExactGPModel(ExactGP):
    """ExactGP用のモデル定義クラス

    ExactGPを使用する場合、本クラスにてモデルを構築する(予定)

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
        self.mean_module = gpytorch.means.ConstantMean()
        _ker_conf = {'ard_num_dims': ex_var_dim}
        _ker_conf.update(ker_conf)
        self.covar_module = set_kernel(kernel, **_ker_conf)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class RunExactGP(object):
    """ExactGPModelの実行クラス

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
    def __init__(self,
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
            random.seed(random_state)
            np.random.seed(random_state)
            torch.manual_seed(random_state)
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
        if self._likelihood in {'GaussianLikelihood', 'GL'}:
            return GaussianLikelihood().to(self.device)
        else:
            raise ValueError

    def _set_mll(self, mll_conf):
        """mllとしてself._mllの指示の元、インスタンスを立てるメソッド
        """
        # mllのインスタンスを立てる
        if self._mll == 'ExactMarginalLogLikelihood':
            return gpytorch.mlls.ExactMarginalLogLikelihood(
                self.likelihood,
                self.model
            )
        else:
            raise ValueError

    def _set_optimizer(self, lr, opt_conf):
        """optimizerとしてself._optimizerの指示の元、インスタンスを立てるメソッド
        """
        if self._optimizer == 'Adam':
            return torch.optim.Adam([
                {'params': self.model.parameters()}
            ], lr=lr, **opt_conf)
        elif self._optimizer == 'SGD':
            return torch.optim.SGD([
                {'params': self.model.parameters()}
            ], lr=lr, **opt_conf)
        elif self._optimizer == 'RMSprop':
            return torch.optim.RMSprop([
                {'params': self.model.parameters()}
            ], lr=lr, **opt_conf)
        elif self._optimizer == 'Adadelta':
            return torch.optim.Adadelta([
                {'params': self.model.parameters()}
            ], lr=lr, **opt_conf)
        elif self._optimizer == 'Adagrad':
            return torch.optim.Adagrad([
                {'params': self.model.parameters()}
            ], lr=lr, **opt_conf)
        else:
            raise ValueError

    def set_model(self,
                  train_x,
                  train_y,
                  lr=1e-3,
                  kernel=None,
                  ard_option=None,
                  ker_conf=None,
                  opt_conf=None):
        """使用するモデルのインスタンスを立てるメソッド

        Parameters
        ----------
        train_x : np.array or torch.tensor
            学習用データセットの説明変数
        train_y : np.array or torch.tensor
            学習用データセットの目的変数
        lr : float
            学習率
        kernel : str or :obj:`gpytorch.kernels`, default 'RBFKernel
            使用するカーネル関数を指定する

            基本はstrで指定されることを想定しているものの、自作のカーネル関数を入力することも可能
        ard_option : bool, default None
            ARDカーネルを利用するかが指定される
        ker_conf : dict, default dict()
            kernelに渡す設定一覧辞書
        opt_conf : dict, default dict()
            optimizerに渡す設定一覧辞書
        """
        if type(train_x) == np.ndarray:
            train_x = array_to_tensor(train_x)
        if type(train_y) == np.ndarray:
            train_y = array_to_tensor(train_y)
        if ard_option is None:
            ard_option = self.ard_option
            ex_var_dim = train_x.shape[1]

        if kernel is None:
            kernel = self._kernel
        if ker_conf is None:
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
        if opt_conf is None:
            opt_conf = self._opt_conf
        self.optimizer = self._set_optimizer(lr, opt_conf=opt_conf)

    def fit(self, epochs, test_x=None, test_y=None, verbose=True):
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
        if type(test_x) == np.ndarray:
            test_x = array_to_tensor(test_x)
        if type(test_y) == np.ndarray:
            test_y = array_to_tensor(test_y)

        for epoch in range(epochs):
            train_loss = []
            test_loss = []
            self.model.train()
            self.likelihood.train()
            self.optimizer.zero_grad()
            # 今回のみの処理な気がする([0]のところ)
            output = self.model(self.model.train_inputs[0])
            loss = - self.mll(output, self.model.train_targets)
            loss.backward()
            self.optimizer.step()

            self.loss.append(loss.item())
            train_loss.append(loss.item())

            self.model.eval()
            self.likelihood.eval()
            if test_x is not None and test_y is not None:
                with torch.no_grad():
                    output = self.model(test_x)
                    loss = - self.mll(output, test_y)
                    test_loss.append(loss.item())

            if epoch % (epochs//10) == 0 and verbose:
                if test_loss:
                    print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {np.mean(train_loss):.3f} / Test Loss: {np.mean(test_loss):.3f}')
                else:
                    print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {np.mean(train_loss):.3f}')
        # TODO: 追加学習のために再学習の際、self.epochを利用する形にする
        self.epoch = epoch + 1

    def predict(self, X):
        """予測用メソッド

        Parameters
        ----------
        X : np.array or torch.tensor
            入力説明変数

        Returns
        -------
        predicts : :obj:`gpytorch.distributions.multivariate_normal.MultivariateNormal`
            予測された目的変数のオブジェクト

            likelihoodの__call__が呼び出されており、平均・標準偏差意外にも多くの要素で構成されている。
        predicts_mean : np.array
            予測された目的変数の平均値
        predicts_std : np.array
            予測された目的変数の標準偏差(0.5 sigma?)
        """
        if type(X) == np.ndarray:
            X = array_to_tensor(X)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            predicts = self.likelihood(self.model(X))
            predicts_mean = tensor_to_array(predicts.mean)
            predicts_std = tensor_to_array(predicts.stddev)
        return predicts, (predicts_mean, predicts_std)

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
        self.epoch, self.model, self.likelihood, self.mll, self.optimizer, self.loss = load_model(file_path, **data)

    # TODO : def kernel_coeff(self)

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

    def plot(self):
        # TODO: 何が必要か定めて、実装
        pass

    def get_params(self) -> dict:
        # TODO: 何を出力するか定めて、実装
        pass


def main():
    # GPでは入力は多次元前提なので (num_data, dim) という shape
    # 一方で出力は一次元前提なので (num_data) という形式にする
    train_inputs = np.linspace(0, 1, 10).reshape(10, 1)
    train_targets = np.sin(2*np.pi*train_inputs).reshape(10) \
        + 0.3*np.random.randn(10)
    test_x = np.linspace(-0.2, 1.2, 100)
    test_y = np.sin(2*np.pi*test_x).reshape(100) \
        + 0.3*np.random.randn(100)

    run = RunExactGP()
    run.set_model(train_inputs, train_targets)
    run.fit(2000, test_x, test_y, verbose=True)
    run.save('test.pth')        # モデルをsave
    run.load('test.pth')        # モデルをload

    predicts, (predicts_mean, predicts_std) = run.predict(test_x)

    # plotはまだ未実装
    plt.style.use('seaborn-darkgrid')
    plt.plot(test_x, predicts_mean)
    plt.plot(test_x, predicts_mean - predicts_std, color='orange')
    plt.plot(test_x, predicts_mean + predicts_std, color='orange')
    plt.fill_between(
        test_x,
        predicts_mean - predicts_std,
        predicts_mean + predicts_std,
        alpha=0.4
    )
    plt.plot(train_inputs, train_targets, "ro")
    plt.show()


if __name__ == '__main__':
    main()
