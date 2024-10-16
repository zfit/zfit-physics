from __future__ import annotations

from typing import TYPE_CHECKING

import zfit

if TYPE_CHECKING:
    try:
        import ROOT
    except ImportError:
        ROOT = None


def roo2z_param(v: ROOT.RooRealVar) -> zfit.Parameter:
    """
    Converts a RooFit RooRealVar to a zfit parameter.

    Args:
        v: RooFit RooRealVar to convert.

    Returns:
        A zfit.Parameter object with properties copied from the RooFit variable.
    """

    name = v.GetName()
    value = v.getVal()
    label = v.GetTitle()
    lower = v.getMin()
    upper = v.getMax()
    floating = not v.isConstant()
    stepsize = None
    if v.hasError():
        stepsize = v.getError()
    elif v.hasAsymError():  # just take average
        stepsize = (v.getErrorHi() - v.getErrorLo()) / 2
    return zfit.Parameter(name, value, lower=lower, upper=upper, floating=floating, step_size=stepsize, label=label)
