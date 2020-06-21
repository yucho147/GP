from math import floor

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import matplotlib.pyplot as plt
import numpy as np
import torch

from gp.utils.utils import (check_device,
                            tensor_to_array,
                            array_to_tensor,
                            load_model,
                            save_model)



class ApproximateGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(ApproximateGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class RunApproximateGP(object):
    def __init__(self,
                 inducing_points_num=0.5,
                 likelihood='GaussianLikelihood',
                 optimizer='Adam',
                 mll='VariationalELBO'):
        self.device = check_device()
        self.inducing_points_num = inducing_points_num
        self._likelihood = likelihood
        self._set_likelihood()
        self._optimizer = optimizer
        self._mll = mll
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
                  lr=1e-3):
        if type(train_x) == np.ndarray:
            train_x = array_to_tensor(train_x)
        if type(train_y) == np.ndarray:
            train_y = array_to_tensor(train_y)
        if isinstance(self.inducing_points_num, float) and self.inducing_points_num < 1:
            inducing_points_len = int(len(train_x) * self.inducing_points_num)
            inducing_points = train_x[:inducing_points_len, :]
        elif isinstance(self.inducing_points_num, int):
            inducing_points_len = self.inducing_points_num
            inducing_points = train_x[:inducing_points_len, :]
        else:
            raise ValueError
        # ここで上記モデルのインスタンスを立てる
        self.model = ApproximateGPModel(
            inducing_points
        ).to(self.device)
        if self._mll == 'VariationalELBO':
            # ここで上記周辺化のインスタンスを立てる
            self.mll = gpytorch.mlls.VariationalELBO(
                self.likelihood,
                self.model,
                num_data=train_y.size(0)
            )
            if self._optimizer == 'Adam':
                # ここで損失関数のインスタンスを立てる
                self.optimizer = torch.optim.Adam([
                    {'params': self.model.parameters()},
                    {'params': self.likelihood.parameters()}
                ], lr=lr)
            else:
                raise ValueError
        else:
            raise ValueError

    def fit(self, epochs, dataloader, verbose=True):
        self.model.train()
        self.likelihood.train()
        for epoch in range(epochs):
            for x_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                output = self.model(x_batch)
                loss = - self.mll(output, y_batch)
                loss.backward()
                self.optimizer.step()

                self.loss.append(loss.item())

            if epoch % (epochs//10) == 0 and verbose:
                print(f'Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.3f}')
        # TODO: 追加学習のために再学習の際、self.epochを利用する形にする
        self.epoch = epoch + 1

    def predict(self, X):
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
        data = dict(
            epoch=self.epoch,
            model=self.model,
            likelihood=self.likelihood,
            mll=self.mll,
            optimizer=self.optimizer,
            loss=self.loss
        )
        self.epoch, self.model, self.likelihood, self.mll, self.optimizer, self.loss = load_model(file_path, **data)

    def plot(self):
        # TODO: 何が必要か定めて、実装
        pass

    def get_params(self) -> dict:
        # TODO: 何を出力するか定めて、実装
        pass


def main():
    # GPでは入力は多次元前提なので (num_data, dim) という shape
    # 一方で出力は一次元前提なので (num_data) という形式にする
    num = 3500
    date_time = np.arange(num)
    input_1 = np.sin(np.arange(num) * 0.05) + np.random.randn(num) / 6
    input_2 = np.sin(np.arange(num) * 0.05 / 1.5) + input_1 + np.random.randn(num) / 6
    input_3 = np.cos(np.arange(num) * 0.05 / 2) + input_2
    input_3 = input_3 + np.random.randn(num) / 2 * input_3  # 自分自身の値が大きいと誤差項が大きくなるように設定
    step = 30
    data = np.array([date_time[:-step], input_1[:-step], input_2[:-step], input_3[:-step], input_3[step:]]).T

    train_n = int(floor(0.9 * len(data)))
    train_inputs = data[:train_n, 1:-1]
    train_targets = data[:train_n, -1]

    test_inputs = data[train_n:, 1:-1]
    test_targets = data[train_n:, -1]

    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(array_to_tensor(train_inputs), array_to_tensor(train_targets))
    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)

    run = RunApproximateGP()
    run.set_model(train_inputs, train_targets, lr=3e-2)
    run.fit(15, train_loader, verbose=True)
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
