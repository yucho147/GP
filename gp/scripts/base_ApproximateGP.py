from math import floor

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


from gp.utils.utils import (check_device,
                            tensor_to_array,
                            array_to_tensor,
                            load_model,
                            save_model)


class ApproximateGPModel(ApproximateGP):
    """ApproximateGP用のモデル定義クラス

    ApproximateGPを使用する場合、本クラスにてモデルを構築する(予定)

    Parameters
    ----------
    inducing_points : torch.tensor
        補助変数の座標

        `learn_inducing_locations=True` である以上、ここで指定する補助変数は更新される
    ex_var_dim : int
        説明変数の個数

        `ex_var_dim=None` を指定すると計算は速くなるものの、説明変数ごとの重みの縮退はとけない。
        結果、一般的に精度は落ちることが考えられる。
    """
    def __init__(self, inducing_points, ex_var_dim):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super(ApproximateGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ex_var_dim)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class RunApproximateGP(object):
    """ApproximateGPModelの実行クラス

    ApproximateGPModelをラップし、学習・予測・プロット等を司る

    Parameters
    ----------
    inducing_points_num : int or float
        補助変数の個数(int)

        もし 0 < inducing_points_num < 1 が渡された場合学習用データの len と inducing_points_num の積が補助変数の個数として設定される
    likelihood : str, default 'GaussianLikelihood'
        likelihoodとして使用するクラス名が指定される
    optimizer : str, default 'Adam'
        optimizerとして使用するクラス名が指定される
    mll : str, default 'VariationalELBO'
        確率分布の周辺化の方法のクラス名が指定される
    ard_option : bool, default True
        ARDカーネルを利用するかが指定される

        もし :obj:`RunApproximateGP.kernel_coeff` を利用する場合 `ard_option=True` を選択する
    """
    def __init__(self,
                 inducing_points_num=0.5,
                 likelihood='GaussianLikelihood',
                 optimizer='Adam',
                 mll='VariationalELBO',
                 ard_option=True):
        self.device = check_device()
        self.inducing_points_num = inducing_points_num
        self._likelihood = likelihood
        self._set_likelihood()
        self._optimizer = optimizer
        self._mll = mll
        self.ard_option = ard_option
        self.epoch = 0
        self.model = None  # 空のmodelを作成しないとloadできない
        self.mll = None    # 空のmodelを作成しないとloadできない
        self.optimizer = None  # 空のmodelを作成しないとloadできない
        self.loss = []

    def _set_likelihood(self):
        """likelihoodとしてself._likelihoodの指示の元、インスタンスを立てるメソッド
        """
        if self._likelihood == 'GaussianLikelihood':
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        else:
            raise ValueError

    def set_model(self,
                  train_x,
                  train_y,
                  lr=1e-3,
                  batch_size=128,
                  shuffle=True,
                  ard_option=None):
        """使用するモデルのインスタンスを立てるメソッド

        Parameters
        ----------
        train_x : np.array or torch.tensor
            学習用データセットの説明変数
        train_y : np.array or torch.tensor
            学習用データセットの目的変数
        lr : float
            学習率
        batch_size : int, default 128
            バッチ数
        shffle : bool, default True
            学習データをシャッフルしてミニバッチ学習させるかを設定
        ard_option : bool, default None
            ARDカーネルを利用するかが指定される
        """
        if type(train_x) == np.ndarray:
            train_x = array_to_tensor(train_x)
        if type(train_y) == np.ndarray:
            train_y = array_to_tensor(train_y)
        if isinstance(self.inducing_points_num, float) and \
           self.inducing_points_num < 1:
            inducing_points_len = int(len(train_x) * self.inducing_points_num)
            indices = torch.randperm(len(train_x))[:inducing_points_len]
            inducing_points = train_x[indices]
        elif isinstance(self.inducing_points_num, int):
            inducing_points_len = self.inducing_points_num
            indices = torch.randperm(len(train_x))[:inducing_points_len]
            inducing_points = train_x[indices]
        else:
            raise ValueError
        if ard_option is None:
            ard_option = self.ard_option
            ex_var_dim = train_x.shape[1]

        train_dataset = TensorDataset(train_x, train_y)
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle)

        # ここで上記モデルのインスタンスを立てる
        if ard_option:
            self.model = ApproximateGPModel(
                inducing_points,
                ex_var_dim=ex_var_dim
            ).to(self.device)
        else:
            self.model = ApproximateGPModel(
                inducing_points,
                ex_var_dim=None
            ).to(self.device)

        # mllのインスタンスを立てる
        if self._mll in {'VariationalELBO', 'VELBO'}:
            self.mll = gpytorch.mlls.VariationalELBO(
                self.likelihood,
                self.model,
                num_data=train_y.size(0)
            )
        elif self._mll in {'PredictiveLogLikelihood', 'PLL'}:
            self.mll = gpytorch.mlls.PredictiveLogLikelihood(
                self.likelihood,
                self.model,
                num_data=train_y.size(0)
            )
        elif self._mll in {'GammaRobustVariationalELBO', 'GRVELBO'}:
            self.mll = gpytorch.mlls.GammaRobustVariationalELBO(
                self.likelihood,
                self.model,
                num_data=train_y.size(0)
            )
        else:
            raise ValueError

        # optimizerのインスタンスを立てる
        if self._optimizer == 'Adam':
            self.optimizer = torch.optim.Adam([
                {'params': self.model.parameters()},
                {'params': self.likelihood.parameters()}
            ], lr=lr)
        else:
            raise ValueError

    def fit(self,
            epochs,
            *,
            train_dataloader=None,
            test_dataloader=None,
            verbose=True):
        """学習用メソッド

        Parameters
        ----------
        epochs : int
            エポック数
        train_dataloader : :obj:`torch.utils.data.DataLoader`, default None
            学習データをまとめたデータローダー
        test_dataloader : :obj:`torch.utils.data.DataLoader`, default None
            テストデータをまとめたデータローダー

            もし test_dataloader を設定している場合エポックごとにテストデータに対するlossも表示されるように設定される
        verbose : bool, default True
            表示形式
        """
        if train_dataloader is None:
            train_dataloader = self.train_loader
        for epoch in range(epochs):
            train_loss = []
            test_loss = []
            self.model.train()
            self.likelihood.train()
            for x_batch, y_batch in train_dataloader:
                self.optimizer.zero_grad()
                output = self.model(x_batch)
                loss = - self.mll(output, y_batch)
                loss.backward()
                self.optimizer.step()

                self.loss.append(loss.item())
                train_loss.append(loss.item())

            self.model.eval()
            self.likelihood.eval()
            if test_dataloader is not None:
                for x_batch, y_batch in test_dataloader:
                    with torch.no_grad():
                        output = self.model(x_batch)
                        loss = - self.mll(output, y_batch)
                        test_loss.append(loss.item())

            if epoch % (epochs//10) == 0 and verbose:
                if test_loss:
                    print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {np.mean(train_loss):.3f} / Test Loss: {np.mean(test_loss):.3f}')
                else:
                    print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {np.mean(train_loss):.3f}')
        # TODO: 追加学習のために再学習の際、self.epochを利用する形にする
        self.epoch = epoch + 1

    def predict(self, X):
        """予測用メソッド

        Parameters
        ----------
        X : np.array or torch.tensor
            入力説明変数

        Returns
        -------
        predicts : :obj:`gpytorch.distributions.multivariate_normal.MultivariateNormal`
            予測された目的変数のオブジェクト

            likelihoodの__call__が呼び出されており、平均・標準偏差以外にも多くの要素で構成されている。
        predicts_mean : np.array
            予測された目的変数の平均値
        predicts_std : np.array
            予測された目的変数の標準偏差(1 sigma)
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
        """モデルのsaveメソッド

        Parameters
        ----------
        file_path : str
            モデルの保存先のパスとファイル名
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
        """モデルのloadメソッド

        Parameters
        ----------
        file_path : str
            モデルの保存先のパスとファイル名
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

    def kernel_coeff(self):
        """kernelの係数を出力するメソッド

        Returns
        -------
        output_dict : dict
            カーネル関数の係数

            `ard_option=True` の場合、 $\Theta$ が各々の説明変数ごとに重みを変えて更新され、出力される

        Warning
        --------
        RBFKernelの場合、各説明変数の重要度 $\eta$ は出力される `'base_kernel.raw_lengthscale'` の逆数の2乗に対応する
        """
        # TODO: kernel関数をスイッチさせ、それに応じてわかりやすい形に変形する
        output_dict = self.model.covar_module.state_dict()
        return output_dict

    def plot(self):
        # TODO: 何が必要か定めて、実装
        pass

    def get_params(self) -> dict:
        # TODO: 何を出力するか定めて、実装
        pass


def main():
    num = 1500
    date_time = np.arange(num)
    input_1 = np.sin(np.arange(num) * 0.05) + np.random.randn(num) / 6
    input_2 = np.sin(np.arange(num) * 0.05 / 1.5) + input_1 + np.random.randn(num) / 6
    input_3 = np.cos(np.arange(num) * 0.05 / 2) + input_2
    input_3 = input_3 + np.random.randn(num) / 2 * input_3  # 自分自身の値が大きいと誤差項が大きくなるように設定
    step = 10
    data = np.array([date_time[:-step], input_1[:-step], input_2[:-step], input_3[:-step], input_3[step:]]).T

    train_n = int(floor(0.9 * len(data)))
    train_inputs = data[:train_n, 1:-1]
    train_targets = data[:train_n, -1]

    test_inputs = data[train_n:, 1:-1]
    test_targets = data[train_n:, -1]

    test_dataset = TensorDataset(array_to_tensor(test_inputs),
                                 array_to_tensor(test_targets))
    test_loader = DataLoader(test_dataset,
                             batch_size=500)

    run = RunApproximateGP(mll='PredictiveLogLikelihood')
    run.set_model(train_inputs, train_targets, lr=3e-2, batch_size=10)
    run.fit(10, test_dataloader=test_loader, verbose=True)
    # test_dataloaderにDataLoaderを渡せば、val lossも出力されるようになる
    # もしない場合にはtrainのlossのみが出力される
    run.save('test.pth')        # モデルをsave
    run.load('test.pth')        # モデルをload

    predicts, (predicts_mean, predicts_std) = run.predict(test_inputs)

    # plotはまだ未実装
    plt.style.use('seaborn-darkgrid')
    plt.plot(data[train_n:, 0], predicts_mean, label='predicts', linewidth=2)
    plt.plot(data[train_n:, 0], test_targets, label='true value', linewidth=1)
    plt.fill_between(
        data[train_n:, 0],
        predicts_mean - predicts_std,
        predicts_mean + predicts_std,
        alpha=0.6,
        label='var'
    )
    plt.title(f'step={step}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
