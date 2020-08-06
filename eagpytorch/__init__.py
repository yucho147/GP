#!/usr/bin/env python3

__version__ = '0.1.0'

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
