#!/usr/bin/env python3
from pkg_resources import get_distribution

__version__ = get_distribution('eagpytorch').version

__all__ = [
    "ApproximateGPModel",
    "ExactGPModel",
    "RunApproximateGP",
    "RunExactGP",
    "utils"
]

from ._base_ApproximateGP import (
    ApproximateGPModel,
    RunApproximateGP
)
from ._base_ExactGP import (
    ExactGPModel,
    RunExactGP
)
from . import utils
