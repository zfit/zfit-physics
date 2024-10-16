from __future__ import annotations

import numpy as np
import zfit
from zfit.util.container import convert_to_container


def nll_from_pyhf(data, pdf, *, params=None, init_pars=None, par_bounds=None, fixed_params=None, errordef=None):
    """Create a zfit loss function from a pyhf pdf and data.


    Args:
        data: Data to be used in the negative log-likelihood (NLL) calculation.
        pdf: Probability density function model.
        params: Optional; Model parameters for the fit. If None, initial parameters, bounds, and fixed status are recommended by the pdf.
        init_pars: Optional; Initial parameter values. Ignored if params is provided.
        par_bounds: Optional; Parameter bounds. Ignored if params is provided.
        fixed_params: Optional; Boolean indicators of parameter fixed statuses. Ignored if params is provided.
        errordef: Optional; Error definition, by default set to 0.5.
    """
    if params is None:
        init_pars = init_pars or pdf.config.suggested_init()
        par_bounds = par_bounds or pdf.config.suggested_bounds()
        fixed_params = fixed_params or pdf.config.suggested_fixed()

        from pyhf.infer import mle

        mle._validate_fit_inputs(init_pars, par_bounds, fixed_params)

        params = [
            zfit.Parameter(f"param_{i}", init, bound[0], bound[1], floating=not is_fixed)
            for i, (init, is_fixed, bound) in enumerate(zip(init_pars, fixed_params, par_bounds))
        ]
    else:
        if init_pars is not None or par_bounds is not None or fixed_params is not None:
            msg = "If `params` are given, `init_pars`, `par_bounds` and `fixed_params` must be None."
            raise ValueError(msg)
        params = convert_to_container(params)

    if errordef is None:
        errordef = 0.5

    def nll_func(params, *, data=data, pdf=pdf, errordef=errordef):
        params = np.asarray(params)
        return mle.twice_nll(params, data, pdf) * errordef

    return zfit.loss.SimpleLoss(func=nll_func, params=params, errordef=errordef)
