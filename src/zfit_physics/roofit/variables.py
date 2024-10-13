from __future__ import annotations


def roo2z_param(v):
    """
    Converts a RooFit RooRealVar to a zfit parameter.

    Args:
        v: RooFit RooRealVar to convert.

    Returns:
        A zfit.Parameter object with properties copied from the RooFit variable.
    """
    import zfit

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
