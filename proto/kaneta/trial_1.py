import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch




class GPModel(gpytorch.models.ExactGP):
    """Documentation for GPModel

    """
    def __init__(self, train_x, train_y, likelihood,
                 lengthscale_prior=None, outputscale_prior=None):
        super(GPModel, self).__init__(train_x,
                                      train_y,
                                      likelihood,
                                      )
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior),
            outputscale_prior=outputscale_prior
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class Trainer(object):
    """Documentation for Trainer

    """
    def __init__(self, gpr, likelihood, optimizer, mll):
        self.gpr = gpr
        self.likelihood = likelihood
        self.optimizer = optimizer
        self.mll = mll

    def update_hyperparameter(self, epochs):
        self.gpr.train()
        self.likelihood.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.gpr(self.gpr.train_inputs[0])

            loss = - self.mll(output, self.gpr.train_targets)
            loss.backward()
            self.optimizer.step()

            if epoch % (epochs//10) == 0:
                print(f'Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.3f}')


def main():
    ## GPでは入力は多次元前提なので (num_data, dim) という shape
    ## 一方で出力は一次元前提なので (num_data) という形式にする
    train_inputs = torch.linspace(0, 1, 10).reshape(10, 1)
    train_targets = torch.sin(2*np.pi*train_inputs).reshape(10) + 0.3*torch.randn(10)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    l_prior = gpytorch.priors.NormalPrior(loc=torch.tensor(1.),
                                          scale=torch.tensor(10.))
    s_prior = gpytorch.priors.NormalPrior(loc=torch.tensor(1.),
                                          scale=torch.tensor(10.))
    gpr = GPModel(train_inputs, train_targets, likelihood,
                  lengthscale_prior=l_prior, outputscale_prior=s_prior)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gpr)

    optimizer = torch.optim.RMSprop(params=gpr.parameters(),
                                    lr=1e-2)
    trainer = Trainer(gpr, likelihood, optimizer, mll)
    trainer.update_hyperparameter(2000)

    test_inputs = torch.linspace(-0.2, 1.2, 100)

    gpr.eval()
    likelihood.eval()
    with torch.no_grad():
        predicts = likelihood(gpr(test_inputs))
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
