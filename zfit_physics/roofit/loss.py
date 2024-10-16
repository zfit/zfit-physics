#  Copyright (c) 2024 zfit
from __future__ import annotations

from contextlib import suppress

import zfit
from zfit.util.container import convert_to_container

from .variables import roo2z_param


def nll_from_roofit(nll, params=None):
    """
    Converts a RooFit NLL (negative log-likelihood) to a Zfit loss object.

    Args:
        nll: The RooFit NLL object to be converted.

    Returns:
        zfit.loss.SimpleLoss: The converted Zfit loss object.

    Raises:
        TypeError: If the provided RooFit loss does not have an error level.
    """
    params = {} if params is None else {p.name: p for p in convert_to_container(params)}

    ROOT = None
    if "cppyy.gbl.RooAbsReal" in str(type(nll)):
        with suppress(ImportError):
            import ROOT
    if ROOT is None or not isinstance(nll, ROOT.RooAbsReal):
        return False  # not a RooFit loss

    import zfit

    def roofit_eval(x):
        for par, arg in zip(nll.getVariables(), x):
            par.setVal(arg)
        # following RooMinimizerFcn.cxx
        nll.setHideOffset(False)
        r = nll.getVal()
        nll.setHideOffset(True)
        return r

    paramsall = []
    for v in nll.getVariables():
        param = params[name] if (name := v.GetName()) in params else roo2z_param(v)
        paramsall.append(param)

    if (errordef := getattr(nll, "defaultErrorLevel", lambda: None)()) is None and (
        errordef := getattr(nll, "errordef", lambda: None)()
    ) is None:
        msg = (
            "Provided loss is RooFit loss but has not error level. "
            "Either set it or create an attribute on the fly (like `nllroofit.errordef = 0.5` "
        )
        raise TypeError(msg)
    return zfit.loss.SimpleLoss(roofit_eval, paramsall, errordef=errordef, jit=False, gradient="num", hessian="num")


zfit.loss.SimpleLoss.register_convertable_loss(nll_from_roofit, priority=50)
