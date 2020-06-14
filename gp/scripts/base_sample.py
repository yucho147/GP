import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch


class ExactGPModel(gpytorch.models.ExactGP):
    """ExactGP用のモデル定義クラス

    ExactGPを使用する場合、本クラスにてモデルを構築する(予定)

    Attributes
    ----------
    mean_module : :obj:`gpytorch.means.ConstantMean`
        使用する平均推定用のインスタンス
    covar_module : :obj:`gpytorch.kernels.ScaleKernel`
        使用するカーネル関数のインスタンス

    Parameters
    ----------
    train_x : torch.tensor
        学習用データセットの説明変数
    train_y : torch.tensor
        学習用データセットの目的変数
    likelihood : :obj:`gpytorch.likelihoods`
        likelihoodのインスタンス
    lengthscale_prior : :obj:`gpytorch.priors.NormalPrior`, default None
        RBFKernelのexpの肩の分母を指すパラメータ

        どの程度離れた点の影響を考慮するかを調整するパラメータとなる
    outputscale_prior : :obj:`gpytorch.priors.NormalPrior`, default None
        RBFKernelの係数を指すパラメータ
    """
    def __init__(self,
                 train_x,
                 train_y,
                 likelihood,
                 lengthscale_prior=None,
                 outputscale_prior=None):
        super(ExactGPModel, self).__init__(train_x,
                                           train_y,
                                           likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior),
            outputscale_prior=outputscale_prior
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class RunExactGP(object):
    """ExactGPModelの実行クラス

    ExactGPModelをラップし、学習・予測・プロット等を司る

    Attributes
    ----------
    l_prior : :obj:`gpytorch.priors.NormalPrior`
        RBFKernelのexpの肩の分母を指すパラメータ

        どの程度離れた点の影響を考慮するかを調整するパラメータとなる
    s_prior : :obj:`gpytorch.priors.NormalPrior`
        RBFKernelの係数を指すパラメータ
    _likelihood : str
        likelihoodを指定する文字列
    likelihood : :obj:`gpytorch.likelihoods`
        likelihoodのインスタンス
    _optimizer : str
        optimizerを指定する文字列
    optimizer : :obj:`torch.optim`
        optimizerのインスタンス
    _mll : str
        確率分布の周辺化の方法を指定する文字列
    mll : :obj:`gpytorch.mlls`
        確率分布の周辺化のインスタンス
    model : :obj:`gpytorch.models`
        ガウス過程のモデルのインスタンス

    Parameters
    ----------
    l_prior : :obj:`gpytorch.priors.NormalPrior`
        RBFKernelのexpの肩の分母を指すパラメータ

        どの程度離れた点の影響を考慮するかを調整するパラメータとなる
    s_prior : :obj:`gpytorch.priors.NormalPrior`
        RBFKernelの係数を指すパラメータ
    likelihood : str
        likelihoodとして使用するクラス名が指定される
    optimizer : str
        optimizerとして使用するクラス名が指定される
    mll : str
        確率分布の周辺化の方法のクラス名が指定される
    """
    def __init__(self,
                 l_prior=gpytorch.priors.NormalPrior(loc=torch.tensor(1.), scale=torch.tensor(10.)),  # 要検討
                 s_prior=gpytorch.priors.NormalPrior(loc=torch.tensor(1.), scale=torch.tensor(10.)),  # 要検討
                 likelihood='GaussianLikelihood',
                 optimizer='RMSprop',
                 mll='ExactMarginalLogLikelihood'):
        self.l_prior = l_prior
        self.s_prior = s_prior
        self._likelihood = likelihood
        self._set_likelihood()
        self._optimizer = optimizer
        self._mll = mll

    def _set_likelihood(self):
        """likelihoodとしてself._likelihoodの指示の元、インスタンスを立てるメソッド
        """
        if self._likelihood == 'GaussianLikelihood':
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        else:
            raise ValueError

    def set_model(self,
                  train_x,
                  train_y,
                  lr=1e-3):
        """使用するモデルのインスタンスを立てるメソッド

        Parameters
        ----------
        train_x : torch.tensor
            学習用データセットの説明変数
        train_y : torch.tensor
            学習用データセットの目的変数
        lr : float
            学習率
        """
        # ここで上記モデルのインスタンスを立てる
        self.model = ExactGPModel(
            train_x,
            train_y,
            self.likelihood,
            self.l_prior,
            self.s_prior
        )
        if self._mll == 'ExactMarginalLogLikelihood':
            # ここで上記周辺化のインスタンスを立てる
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.likelihood,
                self.model
            )
            if self._optimizer == 'RMSprop':
                # ここで損失関数のインスタンスを立てる
                self.optimizer = torch.optim.RMSprop(
                    params=self.model.parameters(),
                    lr=lr
                )
            else:
                raise ValueError
        else:
            raise ValueError

    def fit(self, epochs, verbose=True):
        """学習用メソッド

        Parameters
        ----------
        epochs : int
            エポック数
        verbose : bool, default True
            表示形式
        """
        self.model.train()
        self.likelihood.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            # 今回のみの処理な気がする([0]のところ)
            output = self.model(self.model.train_inputs[0])
            loss = - self.mll(output, self.model.train_targets)
            loss.backward()
            self.optimizer.step()

            if epoch % (epochs//10) == 0 and verbose:
                print(f'Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.3f}')

    def predict(self, X):
        """予測用メソッド

        # TODO: inputsはnumpyにして、内部でtensor型に変更する

        Parameters
        ----------
        X : torch.tensor
            入力説明変数

        Returns
        -------
        predicts : :obj:`gpytorch.distributions.multivariate_normal.MultivariateNormal`
            予測された目的変数のオブジェクト

            likelihoodの__call__が呼び出されており、平均・標準偏差意外にも多くの要素で構成されている。
        predicts_mean : torch.tensor
            予測された目的変数の平均値
        predicts_std : torch.tensor
            予測された目的変数の標準偏差(0.5 sigma?)
        """
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            predicts = self.likelihood(self.model(X))
            predicts_mean = predicts.mean
            predicts_std = predicts.stddev
        return predicts, (predicts_mean, predicts_std)

    def plot(self):
        # TODO: 何が必要か定めて、実装
        pass

    def get_params(self) -> dict:
        # TODO: 何を出力するか定めて、実装
        pass


def main():
    # GPでは入力は多次元前提なので (num_data, dim) という shape
    # 一方で出力は一次元前提なので (num_data) という形式にする
    train_inputs = torch.linspace(0, 1, 10).reshape(10, 1)
    train_targets = torch.sin(2*np.pi*train_inputs).reshape(10) \
        + 0.3*torch.randn(10)

    run = RunExactGP()
    run.set_model(train_inputs, train_targets)
    run.fit(2000, verbose=True)

    test_inputs = torch.linspace(-0.2, 1.2, 100)
    predicts, (predicts_mean, predicts_std) = run.predict(test_inputs)

    # plotはまだ未実装
    plt.style.use('seaborn-darkgrid')
    plt.plot(test_inputs.numpy(), predicts_mean.numpy())
    plt.plot(test_inputs.numpy(), predicts_mean.numpy() - predicts_std.numpy(), color='orange')
    plt.plot(test_inputs.numpy(), predicts_mean.numpy() + predicts_std.numpy(), color='orange')
    plt.fill_between(
        test_inputs.numpy(),
        predicts_mean.numpy() - predicts_std.numpy(),
        predicts_mean.numpy() + predicts_std.numpy(),
        alpha=0.4
    )
    plt.plot(train_inputs.numpy(), train_targets.numpy(), "ro")
    plt.show()


if __name__ == '__main__':
    main()
