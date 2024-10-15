from __future__ import annotations

import tensorflow as tf
import zfit
from zfit import z
from zfit.util import ztyping
from zfit.z import numpy as znp


@z.function(wraps="tensor")
def erfexp_pdf_func(x, mu, beta, gamma, n):
    """Calculate the ErfExp PDF.

    Args:
        x: value(s) for which the PDF will be calculated.
        mu: Location parameter.
        beta: Scale parameter.
        gamma: Shape parameter.
        n: Shape parameter.

    Returns:
        `tf.Tensor`: The calculated PDF values.

    Notes:
        Implementation from https://gitlab.cern.ch/cms-muonPOG/spark_tnp/-/blob/Spark3/RooErfExp.cc
        The parameters beta and gamma are given in reverse order in this c++ implementation.
    """
    return tf.math.erfc((x - mu) * beta) * znp.exp(-gamma * (znp.power(x, n) - znp.power(mu, n)))


# Note: There is no analytic integral for the ErfExp PDF
# We tried with sympy, Mathematica, Wolfram Alpha and https://www.integral-calculator.com/
# import sympy as sp
#
# # Define symbols
# x, mu, beta, gamma, n = sp.symbols('x mu beta gamma n', real=True)
#
# # Define the function
# func = sp.erfc((x - mu) * beta) * sp.exp(-gamma * (x**n - mu**n))
# sp.integrate(func, x)
class ErfExp(zfit.pdf.BasePDF):
    _N_OBS = 1

    def __init__(
        self,
        mu: ztyping.ParamTypeInput,
        beta: ztyping.ParamTypeInput,
        gamma: ztyping.ParamTypeInput,
        n: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ztyping.ExtendedInputType | None = False,
        norm: ztyping.NormInputType | None = None,
        name: str = "ErfExp",
        label: str | None = None,
    ):
        """ErfExp PDF, the product of a complementary error function and an exponential function.

        Implementation following closely `C++ version of custom RooErfExp.cc <https://gitlab.cern.ch/cms-muonPOG/spark_tnp/-/blob/Spark3/RooErfExp.cc>`_

        .. math:

            f(x; \\mu, \\beta, \\gamma, n) = \\text{erfc}(\\beta (x - \\mu)) \\exp{(-\\gamma (x^n - \\mu^n))}

        Args:
            mu: Location parameter.
            beta: Scale parameter.
            gamma: Shape parameter, scale of exponential term.
            n: Shape parameter, power in exponential term.
            obs: |@doc:pdf.init.obs| Observables of the
               model. This will be used as the default space of the PDF and,
               if not given explicitly, as the normalization range.

               The default space is used for example in the sample method: if no
               sampling limits are given, the default space is used.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               The observables are not equal to the domain as it does not restrict or
               truncate the model outside this range. |@docend:pdf.init.obs|
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: |@doc:pdf.init.name| Name of the PDF.
               Maybe has implications on the serialization and deserialization of the PDF.
               For a human-readable name, use the label. |@docend:pdf.init.name|
            label: |@doc:pdf.init.label| Human-readable name
               or label of
               the PDF for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.label|
        """
        params = {"mu": mu, "beta": beta, "gamma": gamma, "n": n}
        super().__init__(obs=obs, params=params, extended=extended, norm=norm, name=name, label=label)

    @zfit.supports()
    def _unnormalized_pdf(self, x, params):
        mu = params["mu"]
        beta = params["beta"]
        gamma = params["gamma"]
        n = params["n"]
        x = x[0]
        return erfexp_pdf_func(x=x, mu=mu, beta=beta, gamma=gamma, n=n)
