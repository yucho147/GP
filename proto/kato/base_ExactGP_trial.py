from math import floor

import random

import gpytorch
from gpytorch.models import ExactGP
# from gpytorch.variational import CholeskyVariationalDistribution
# from gpytorch.variational import VariationalStrategy
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


from gp.utils.utils import (check_device,
                            tensor_to_array,
                            array_to_tensor,
                            load_model,
                            save_model)

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
    """
    def __init__(self,
                 train_x,
                 train_y,
                 likelihood,
                 ex_var_dim):
        super(ExactGPModel, self).__init__(train_x,
                                           train_y,
                                           likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ex_var_dim)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class RunExactGP(object):
    """ExactGPModelの実行クラス

    ExactGPModelをラップし、学習・予測・プロット等を司る

    Parameters
    ----------
    l_prior : :obj:`gpytorch.priors.NormalPrior`
        RBFKernelのexpの肩の分母を指すパラメータ

        どの程度離れた点の影響を考慮するかを調整するパラメータとなる
    s_prior : :obj:`gpytorch.priors.NormalPrior`
        RBFKernelの係数を指すパラメータ
    likelihood : str
        likelihoodとして使用するクラス名が指定される
    optimizer : str
        optimizerとして使用するクラス名が指定される
    mll : str
        確率分布の周辺化の方法のクラス名が指定される
    ard_option : bool, default True
        ARDカーネルを利用するかが指定される

        もし :obj:`RunApproximateGP.kernel_coeff` を利用する場合 `ard_option=True` を選択する
    """
    def __init__(self,
                 l_prior=gpytorch.priors.NormalPrior(loc=torch.tensor(1.), scale=torch.tensor(10.)),  # 要検討
                 s_prior=gpytorch.priors.NormalPrior(loc=torch.tensor(1.), scale=torch.tensor(10.)),  # 要検討
                 likelihood='GaussianLikelihood',
                 optimizer='RMSprop',
                 mll='ExactMarginalLogLikelihood'):
        self.device = check_device()
        self.l_prior = l_prior
        self.s_prior = s_prior
        self._likelihood = likelihood
        self._set_likelihood()
        self._optimizer = optimizer
        self._mll = mll
        self.ard_option = ard_option
        self.epoch = 0
        self.model = None  # 空のmodelを作成しないとloadできない
        self.mll = None    # 空のmodelを作成しないとloadできない
        self.optimizer = None  # 空のmodelを作成しないとloadできない
        self.loss = []

    def _set_likelihood(self):
        """likelihoodとしてself._likelihoodの指示の元、インスタンスを立てるメソッド
        """
        if self._likelihood == 'GaussianLikelihood':
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        else:
            raise ValueError

    def set_model(self,
                  train_x,
                  train_y,
                  lr=1e-3,
                  ard_option=None):
        """使用するモデルのインスタンスを立てるメソッド

        Parameters
        ----------
        train_x : np.array or torch.tensor
            学習用データセットの説明変数
        train_y : np.array or torch.tensor
            学習用データセットの目的変数
        lr : float
            学習率
        ard_option : bool, default None
            ARDカーネルを利用するかが指定される
        """
        if type(train_x) == np.ndarray:
            train_x = array_to_tensor(train_x)
        if type(train_y) == np.ndarray:
            train_y = array_to_tensor(train_y)
        if ard_option is None:
            ard_option = self.ard_option
            ex_var_dim = train_x.shape[1]
            
        # ここで上記モデルのインスタンスを立てる
        if ard_option:
            self.model = ExactGPModel(
                train_x,
                train_y,
                self.likelihood,
                ex_var_dim=ex_var_dim,
                self.l_prior,
                self.s_prior
            ).to(self.device)
        else:
            self.model = ExactGPModel(
                train_x,
                train_y,
                self.likelihood,
                ex_var_dim=None,
                self.l_prior,
                self.s_prior
            ).to(self.device)
            
        if self._mll == 'ExactMarginalLogLikelihood':
            # ここで上記周辺化のインスタンスを立てる
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.likelihood,
                self.model
            )
        else:
            raise ValueError

        if self._optimizer == 'RMSprop':
            # ここで損失関数のインスタンスを立てる
            self.optimizer = torch.optim.RMSprop(
                params=self.model.parameters(),
                lr=lr
                )
        else:
            raise ValueError

    def fit(self, epochs, test_dataloader=None, verbose=True):
        """学習用メソッド

        Parameters
        ----------
        epochs : int
            エポック数
        test_dataloader : :obj:`torch.utils.data.DataLoader`, default None
            テストデータをまとめたデータローダー

            もし test_dataloader を設定している場合エポックごとにテストデータに対するlossも表示されるように設定される
        verbose : bool, default True
            表示形式
        """
        for epoch in range(epochs):
            # DONE : train_loss, test_loss を別々にする
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

            self.loss.append(loss.item()) # DONE : train_lossに変更
            train_loss.append(loss.item())

            # TODO : test_loss について処理を行う
            self.model.eval()     
            self.likelihood.eval()
            if test_dataloader is not None: # TODO : dataloaderという形式がいるのか？確認
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
            
            # if epoch % (epochs//10) == 0 and verbose:
            #     # test_loss があるかどうかの分岐
            #     print(f'Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.3f}')
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
        
    def plot(self):
        # TODO: 何が必要か定めて、実装
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
    input_3 = input_3 + np.random.randn(num) / 2 * input_3  # 自分自身の値が大きいと誤差項が大きくなるように設定
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

    run = RunExactGP(mll='PredictiveLogLikelihood')
    run.set_model(train_inputs, train_targets, lr=3e-2, batch_size=10)
    run.fit(10, test_dataloader=test_loader, verbose=True)
    # test_dataloaderにDataLoaderを渡せば、val lossも出力されるようになる
    # もしない場合にはtrainのlossのみが出力される
    run.save('test.pth')        # モデルをsave
    run.load('test.pth')        # モデルをload

    predicts, (predicts_mean, predicts_std) = run.predict(test_inputs)

    # plotはまだ未実装
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
