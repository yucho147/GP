from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from pyro import distributions as base_distributions

from . import _OneDimensionalLikelihood


class PoissonLikelihood(_OneDimensionalLikelihood):
    """Posson回帰に用いるLikelihood
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quadrature = GaussHermiteQuadrature1D(num_locs=20)

    def forward(self, function_samples, **kwargs):
        lam = function_samples.exp()
        return base_distributions.Poisson(rate=lam)

    def marginal(self, function_dist, **kwargs):
        lam = function_dist.mean.exp()
        return base_distributions.Poisson(rate=lam)
