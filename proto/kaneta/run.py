import gpytorch
import matplotlib.pyplot as plt
import torch


class Model_ExactGP(gpytorch.models.ExactGP):
    """Documentation for Model_ExactGP
    ExactGP : train_inputs, train_targets, likelihood
        https://gpytorch.readthedocs.io/en/latest/models.html#gpytorch.models.ExactGP
    """
    def __init__(self, train_inputs, train_targets, likelihood, **params):
        super(Model_ExactGP, self).__init__(train_inputs, train_targets, likelihood)
        ...

    def forward(self, x):
        ...
        return gpytorch.distributions.xxx(...)


class Model_ApproximateGP(gpytorch.models.ApproximateGP):
    """Documentation for Model_ApproximateGP
    ApproximateGP : variational_strategy
        https://gpytorch.readthedocs.io/en/latest/models.html#gpytorch.models.PyroGP
    """
    def __init__(self, variational_strategy, **params):
        super(Model_ApproximateGP, self).__init__(variational_strategy)
        ...

    def forward(self, x):
        ...
        return gpytorch.distributions.xxx(...)


class Model_PyroGP(gpytorch.models.PyroGP):
    """Documentation for Model_PyroGP
    PyroGP : variational_strategy, likelihood, num_data, name_prefix='', beta=1.0
        https://gpytorch.readthedocs.io/en/latest/models.html#gpytorch.models.ApproximateGP
    """
    def __init__(self, variational_strategy, likelihood, num_data,
                 name_prefix='', beta=1.0, **params):
        super(Model_PyroGP, self).__init__(variational_strategy, likelihood, num_data,
                                           name_prefix='', beta=1.0)
        ...

    def forward(self, x):
        ...
        return gpytorch.distributions.xxx(...)


class Trainer(object):
    """Documentation for Trainer

    """
    def __init__(self, gp, likelihood, optimizer, mll):
        self.gp = gp
        self.likelihood = likelihood
        self.optimizer = optimizer
        self.mll = mll

    def update_hyperparameter(self, epochs, verbose=1):
        # TODO: eary_stopping
        self.gp.train()
        self.likelihood.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.gp(self.gp.train_inputs[0])  # 多分train_inputs[0]ではダメ

            loss = - self.mll(output, self.gp.train_targets)
            loss.backward()
            self.optimizer.step()

            if epoch % (epochs//10) == 0 and verbose:
                print(f'Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.3f}')

        return self


class Run(object):
    """Documentation for Run

    """
    def __init__(self,
                 mode='ExactGP',
                 lh='GaussianLikelihood',
                 optim='RMSprop',
                 mll='ExactMarginalLogLikelihood'):

        self._model = self._choice_model(mode)
        self._likelihood = self._choice_likelihood(lh)
        self._optimizer = self._choice_optimizer(optim)
        self._mll = mll
        self.fit_done = False

    def _choice_model(self, mode):
        if mode in {'ExactGP', 'gaussian_process'}:
            return Model_ExactGP
        elif mode in {'ApproximateGP', 'xxx'}:
            return Model_ApproximateGP
        elif mode in {'PyroGP', 'VariationalBeyse'}:
            return Model_PyroGP
        else:
            raise TypeError(f'mode={mode} が定義されていない文字列になっている')

    def _choice_likelihood(self, lh):
        if lh in {'GaussianLikelihood'}:
            return gpytorch.likelihoods.GaussianLikelihood
        elif lh in {'Likelihood'}:
            return gpytorch.likelihoods.Likelihood
        elif lh in {'BernoulliLikelihood'}:
            return gpytorch.likelihoods.BernoulliLikelihood
        elif lh in {'SoftmaxLikelihood'}:
            return gpytorch.likelihoods.SoftmaxLikelihood
        else:
            raise TypeError(f'lh={lh} が定義されていない文字列になっている')

    def _choice_optimizer(self, optim):
        if optim in {'RMSprop'}:
            return torch.optim.RMSprop
        elif optim in {''}:
            return torch.optim.xxx
        else:
            raise TypeError(f'optim={optim} が定義されていない文字列になっている')

    def _choice_mll(self, mll):
        if mll in {'ExactMarginalLogLikelihood'}:
            return gpytorch.mlls.ExactMarginalLogLikelihood
        elif mll in {''}:
            return gpytorch.mlls.xxx
        else:
            raise TypeError(f'mll={mll} が定義されていない文字列になっている')

    def fit(self, x, y, lr=1e-4, epochs=2000, verbose=1, **params):
        self.likelihood = self._likelihood(**params)
        self.gp = self._model(x, y, **params)
        self._mll = self._choice_mll(self.mll)
        self.mll = self._mll(self.likelihood, self.gp)

        self.optimizer = self._optimizer(params=self.gp.parameters(),
                                         lr=lr)

        self.trainer = Trainer(self.gp, self.likelihood, self.optimizer, self.mll)
        self.trainer.update_hyperparameter(epochs=epochs, verbose=verbose)

        self.fit_done = True
        return self

    def predict(self, x):
        if not self.fit_done:
            raise               # なんかエラーを吐く。学習していないので。
        pass

    def save_model(self):
        pass

    def load_model(self):
        self.fit_done = True
        pass

    def plot(self):
        pass

    def get_params(self):
        pass
