from typing import Optional

import tensorflow as tf
import zfit
from zfit import z
from zfit.util import ztyping
from zfit.z import numpy as znp


@z.function(wraps="tensor")
def erfexp_pdf_func(x, alpha, beta, gamma, n):
    """Calculate the ErfExp PDF.

    Args:
        x: value(s) for which the PDF will be calculated.
        alpha: Location parameter.
        beta: Scale parameter.
        gamma: Shape parameter.
        n: Shape parameter.

    Returns:
        `tf.Tensor`: The calculated PDF values.

    Notes:
        Implementation from https://gitlab.cern.ch/cms-muonPOG/spark_tnp/-/blob/Spark3/RooErfExp.cc
        The parameters beta and gamma are given in reverse order in this c++ implementation.
    """
    return tf.math.erfc((x - alpha) * beta) * znp.exp(-gamma * (znp.power(x, n) - znp.power(alpha, n)))


# Note: There is no analytic integral for the ErfExp PDF
# We tried with sympy, Mathematica, Wolfram Alpha and https://www.integral-calculator.com/
# import sympy as sp
#
# # Define symbols
# x, alpha, beta, gamma, n = sp.symbols('x alpha beta gamma n', real=True)
#
# # Define the function
# func = sp.erfc((x - alpha) * beta) * sp.exp(-gamma * (x**n - alpha**n))
# sp.integrate(func, x)
class ErfExp(zfit.pdf.BasePDF):
    _N_OBS = 1

    def __init__(
        self,
        alpha: ztyping.ParamTypeInput,
        beta: ztyping.ParamTypeInput,
        gamma: ztyping.ParamTypeInput,
        n: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: Optional[ztyping.ExtendedInputType] = False,
        norm: Optional[ztyping.NormInputType] = None,
        name: str = "ErfExp",
    ):
        """ErfExp PDF, the product of a complementary error function and an exponential function.

        Implementation from https://gitlab.cern.ch/cms-muonPOG/spark_tnp/-/blob/Spark3/RooErfExp.cc

        .. math:

            f(x; \\alpha, \\beta, \\gamma, n) = \\text{erfc}(\\beta (x - \\alpha)) \\exp{(-\\gamma (x^n - \\alpha^n))}

        Args:
            alpha: Location parameter.
            beta: Scale parameter.
            gamma: Shape parameter.
            n: Shape parameter.
            obs: |@doc:pdf.init.obs| Observables of the
               model. This will be used as the default space of the PDF and,
               if not given explicitly, as the normalization range.

               The default space is used for example in the sample method: if no
               sampling limits are given, the default space is used.

               The observables are not equal to the domain as it does not restrict or
               truncate the model outside this range. |@docend:pdf.init.obs|
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: |@doc:pdf.init.name| Human-readable name
               or label of
               the PDF for better identification.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.name|
        """
        params = {"alpha": alpha, "beta": beta, "gamma": gamma, "n": n}
        super().__init__(obs=obs, params=params, extended=extended, norm=norm)

    def _unnormalized_pdf(self, x):
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        gamma = self.params["gamma"]
        n = self.params["n"]
        x = z.unstack_x(x)
        return erfexp_pdf_func(x=x, alpha=alpha, beta=beta, gamma=gamma, n=n)
