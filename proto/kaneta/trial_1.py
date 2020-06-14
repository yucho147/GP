import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                 lengthscale_prior=None, outputscale_prior=None):
        super(GPModel, self).__init__(train_x,
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

class Run(object):
    def __init__(self,
                 gpr='ExactGP',
                 l_prior=gpytorch.priors.NormalPrior(loc=torch.tensor(1.), scale=torch.tensor(10.)),
                 s_prior=gpytorch.priors.NormalPrior(loc=torch.tensor(1.), scale=torch.tensor(10.)),
                 likelihood='GaussianLikelihood',
                 optimizer='RMSprop',
                 mll='ExactMarginalLogLikelihood'):
        self.l_prior = l_prior
        self.s_prior = s_prior
        self._likelihood = likelihood
        self._set_likelihood()
        self._gpr = gpr
        self._optimizer = optimizer
        self._mll = mll

    def _set_likelihood(self):
        if self._likelihood == 'GaussianLikelihood':
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        else:
            raise ValueError

    def set_model(self, train_x, train_y):
        if self._gpr == 'ExactGP':
            # ここで上記モデルのインスタンスを立てる
            self.model = GPModel(train_x, train_y, self.likelihood,
                                 self.l_prior, self.s_prior)
            if self._mll == 'ExactMarginalLogLikelihood':
                self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
                if self._optimizer == 'RMSprop':
                    self.optimizer = torch.optim.RMSprop(params=self.model.parameters(),
                                                         lr=1e-2)
                else:
                    raise ValueError
            else:
                raise ValueError
        else:
            raise ValueError

    def update_hyperparameter(self, epochs):
        self.model.train()
        self.likelihood.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.model(self.model.train_inputs[0])

            loss = - self.mll(output, self.model.train_targets)
            loss.backward()
            self.optimizer.step()

            if epoch % (epochs//10) == 0:
                print(f'Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.3f}')


def main():
    # GPでは入力は多次元前提なので (num_data, dim) という shape
    # 一方で出力は一次元前提なので (num_data) という形式にする
    train_inputs = torch.linspace(0, 1, 10).reshape(10, 1)
    train_targets = torch.sin(2*np.pi*train_inputs).reshape(10) + 0.3*torch.randn(10)

    run = Run()
    run.set_model(train_inputs, train_targets)
    run.update_hyperparameter(2000)

    test_inputs = torch.linspace(-0.2, 1.2, 100)

    # Runクラスではupdate_hyperparameterしかまだ作っていないのでpredictは下記のように直打ちしているっす。
    run.model.eval()
    run.likelihood.eval()
    with torch.no_grad():
        predicts = run.likelihood(run.model(test_inputs))
        predicts_mean = predicts.mean
        predicts_std = predicts.stddev

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
