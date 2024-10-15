from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    import tf_pwa

import zfit
import zfit.z.numpy as znp
from zfit.core.interfaces import ZfitParameter
from zfit.util.container import convert_to_container

from .variables import params_from_fcn

ParamType = Optional[Union[ZfitParameter, Iterable[ZfitParameter]]]


def nll_from_fcn(fcn: tf_pwa.model.FCN, *, params: ParamType = None):
    """Create a zfit loss from a tf_pwa FCN.

    Args:
        fcn: A tf_pwa.FCN
        params: list of zfit.Parameter, optional
            Parameters to use in the loss. If None, all trainable parameters in the FCN are used.

    Returns:
        zfit.loss.SimpleLoss
    """
    params = params_from_fcn(fcn) if params is None else convert_to_container(params, container=list)
    paramnames = tuple(p.name for p in params)

    # something is off here: for the value, we need to pass the parameters as a dict
    # but for the gradient/hesse, we need to pass them as a list
    # TODO: activate if https://github.com/jiangyi15/tf-pwa/pull/153 is merged
    # @z.function(wraps="loss")
    def eval_func(params):
        paramdict = make_paramdict(params)
        return fcn(paramdict)

    # TODO: activate if https://github.com/jiangyi15/tf-pwa/pull/153 is merged
    # @z.function(wraps="loss")
    def eval_grad(params):
        return fcn.nll_grad(params)[1]

    def make_paramdict(params, *, paramnames=paramnames):
        return {p: znp.array(v.value()) for p, v in zip(paramnames, params)}

    return zfit.loss.SimpleLoss(
        func=eval_func,
        params=params,
        errordef=0.5,
        gradient=eval_grad,
        hessian=lambda x: fcn.nll_grad_hessian(x)[2],
        jit=False,
    )


def _nll_from_fcn_or_false(fcn: tf_pwa.model.FCN, *, params: ParamType = None) -> zfit.loss.SimpleLoss | bool:
    try:
        from tf_pwa.model import FCN
    except ImportError:
        return False
    else:
        if isinstance(fcn, FCN):
            return nll_from_fcn(fcn, params=params)
    return False


zfit.loss.SimpleLoss.register_convertable_loss(_nll_from_fcn_or_false, priority=50)
# Maybe add actually a custom loss?
# class TFPWALoss(zfit.loss.BaseLoss):
#     def __init__(self, loss, params=None):
#         if params is None:
#             params =  [zfit.Parameter(n, v) for n, v in amp.get_params().items() if n in fcn.vm.trainable_vars]
#         self._lossparams = params
#         super().__init__(model=[], data=[], options={"subtr_const": False}, jit=False)
#         self._errordef = 0.5
#         self._tfpwa_loss = loss
#
#     def _value(self, model, data, fit_range, constraints, log_offset):
#         return self._tfpwa_loss(self._lossparams)
#
#     def _value_gradient(self, params, numgrad, full=None):
#         return self._tfpwa_loss.get_nll_grad(params)
#
#     def _value_gradient_hessian(self, params, hessian, numerical=False, full: bool | None = None):
#         return self._tfpwa_loss.get_nll_grad_hessian(params)
#
#     # below is a small hack as zfit is reworking it's loss currently
#     def _get_params(
#             self,
#             floating: bool | None = True,
#             is_yield: bool | None = None,
#             extract_independent: bool | None = True,
#     ):
#         params = super()._get_params(floating, is_yield, extract_independent)
#         from zfit.core.baseobject import extract_filter_params
#         own_params = extract_filter_params(self._lossparams, floating=floating, extract_independent=extract_independent)
#         return params.union(own_params)
#
#     def create_new(self):
#         raise RuntimeError("Not needed, todo")
#
#     def _loss_func(self,):
#         raise RuntimeError("Not needed, needs new release")
