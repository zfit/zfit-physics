from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import zfit
from zfit.util.container import convert_to_container

from .variables import params_from_intensity

if TYPE_CHECKING:
    from tensorwaves.estimator import Estimator
    from zfit.core.interfaces import ZfitLoss

__all__ = ["nll_from_estimator"]


def nll_from_estimator(estimator: Estimator, *, params=None, errordef=None, numgrad=None) -> ZfitLoss:
    r"""Create a negative log-likelihood function from a tensorwaves estimator.

    Args:
        estimator: An estimator object that computes a scalar loss function.
        params: A list of zfit parameters that the loss function depends on.
        errordef: The error definition of the loss function.
        numgrad: If True, the gradient of the loss function is computed numerically and the ComPWA estimators
            gradient method is not used. Can be useful as not all backends in ComPWA support gradients.

    Returns:
        A zfit loss function that can be used with zfit.

    """
    from tensorwaves.estimator import ChiSquared, UnbinnedNLL

    if params is None:
        classname = estimator.__class__.__name__
        intensity = getattr(estimator, f"_{classname}__function", None)
        if intensity is None:
            msg = f"Could not find intensity function in {estimator}. Maybe the attribute changed?"
            raise ValueError(msg)
        params = params_from_intensity(intensity)
    else:
        params = convert_to_container(params)

    paramnames = [param.name for param in params]

    def func(params):
        paramdict = dict(zip(paramnames, params))
        return estimator(paramdict)

    if numgrad:
        grad = None
    else:

        def grad(params):
            paramdict = dict(zip(paramnames, params))
            return estimator.gradient(paramdict)

    if errordef is None:
        if hasattr(estimator, "errordef"):
            errordef = estimator.errordef
        elif isinstance(estimator, ChiSquared):
            errordef = 1.0
        elif isinstance(estimator, UnbinnedNLL):
            errordef = 0.5
    return zfit.loss.SimpleLoss(func=func, gradient=grad, params=params, errordef=errordef)


def _nll_from_estimator_or_false(estimator: Estimator, *, params=None, errordef=None) -> ZfitLoss | bool:
    if "tensorwaves" in repr(type(estimator)):
        try:
            import tensorwaves as tw
        except ImportError:
            return False
        if not isinstance(estimator, (tw.estimator.ChiSquared, tw.estimator.UnbinnedNLL)):
            warnings.warn(
                "Only ChiSquared and UnbinnedNLL are supported from tensorwaves currently."
                f"TensorWaves is in name of {estimator}, this could be a bug.",
                stacklevel=2,
            )
            return False
        return nll_from_estimator(estimator, params=params, errordef=errordef)
    return None


zfit.loss.SimpleLoss.register_convertable_loss(_nll_from_estimator_or_false)
