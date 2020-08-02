__version__ = '0.1.0'

from .scripts import (
    ApproximateGPModel,
    RunApproximateGP,
    ExactGPModel,
    RunExactGP,
    Regression,
    Classifier
)
from .utils import utils


__all__ = [
    "ApproximateGPModel",
    "RunApproximateGP",
    "ExactGPModel",
    "RunExactGP",
    "Regression",
    "Classifier",
    "utiles"
]
