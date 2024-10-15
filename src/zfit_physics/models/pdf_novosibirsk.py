from __future__ import annotations

import numpy as np
import tensorflow as tf
import zfit
import zfit.z.numpy as znp
from zfit import z
from zfit.core.space import ANY_LOWER, ANY_UPPER, Space
from zfit.util import ztyping


@z.function(wraps="tensor")
def novosibirsk_pdf(x, mu, sigma, lambd):
    """Calculate the Novosibirsk PDF.

    Args:
        x: value(s) for which the PDF will be calculated.
        peak: peak of the distribution.
        width: width of the distribution.
        tail: tail of the distribution.

    Returns:
        `tf.Tensor`: The calculated PDF values.

    Notes:
        Function taken from H. Ikeda et al. NIM A441 (2000), p. 401 (Belle Collaboration)
        Based on code from `ROOT <https://root.cern.ch/doc/master/RooNovosibirsk_8cxx_source.html>`_
    """
    # rename parameters to follow roofit implementation
    peak = mu
    width = sigma
    tail = lambd

    cond1 = znp.less(znp.abs(tail), 1e-7)
    arg = 1.0 - (x - peak) * tail / width

    cond2 = znp.less(arg, 1e-7)
    log_arg = znp.log(arg)
    xi = 2 * np.sqrt(np.log(4.0))

    width_zero = (2.0 / xi) * znp.arcsinh(tail * xi * 0.5)
    width_zero2 = width_zero**2
    exponent = (-0.5 / width_zero2 * log_arg**2) - (width_zero2 * 0.5)
    gauss = znp.exp(-0.5 * ((x - peak) / width) ** 2)

    return znp.where(cond1, gauss, znp.where(cond2, 0.0, znp.exp(exponent)))


def novosibirsk_integral(limits: ztyping.SpaceType, params: dict, model) -> tf.Tensor:
    """Calculates the analytic integral of the Novosibirsk PDF.

    Args:
        limits: An object with attribute limit1d.
        params: A hashmap from which the parameters that defines the PDF will be extracted.
        model: Will be ignored.

    Returns:
        The calculated integral.
    """
    del model
    lower, upper = limits.limit1d
    mu = params["mu"]
    sigma = params["sigma"]
    lambd = params["lambd"]

    return novosibirsk_integral_func(mu=mu, sigma=sigma, lambd=lambd, lower=lower, upper=upper)


@z.function(wraps="tensor")
def novosibirsk_integral_func(mu, sigma, lambd, lower, upper):
    """Calculate the integral of the Novosibirsk PDF.

    Args:
        peak: peak of the distribution.
        width: width of the distribution.
        tail: tail of the distribution.
        lower: lower limit of the integral.
        upper: upper limit of the integral.

    Returns:
        `tf.Tensor`: The calculated integral.

    Notes:
        Based on code from `ROOT <https://root.cern.ch/doc/master/RooNovosibirsk_8cxx_source.html>`_
    """
    # rename parameters to follow roofit implementation
    peak = mu
    width = sigma
    tail = lambd
    sqrt2 = np.sqrt(2)
    sqlog2 = np.sqrt(np.log(2))
    sqlog4 = np.sqrt(np.log(4))
    log4 = np.log(4)
    rootpiby2 = np.sqrt(np.pi / 2)
    sqpibylog2 = np.sqrt(np.pi / np.log(2))

    cond = znp.less(znp.abs(tail), 1e-7)

    xscale = sqrt2 * width
    result_gauss = rootpiby2 * width * (tf.math.erf((upper - peak) / xscale) - tf.math.erf((lower - peak) / xscale))

    log_argument_A = znp.maximum(((peak - lower) * tail + width) / width, 1e-7)
    log_argument_B = znp.maximum(((peak - upper) * tail + width) / width, 1e-7)

    term1 = znp.arcsinh(tail * sqlog4)
    term1_2 = term1**2
    erf_termA = (term1_2 - log4 * znp.log(log_argument_A)) / (2 * term1 * sqlog2)
    erf_termB = (term1_2 - log4 * znp.log(log_argument_B)) / (2 * term1 * sqlog2)

    result_novosibirsk = 0.5 / tail * width * term1 * (tf.math.erf(erf_termB) - tf.math.erf(erf_termA)) * sqpibylog2

    return znp.where(cond, result_gauss, result_novosibirsk)


class Novosibirsk(zfit.pdf.BasePDF):
    _N_OBS = 1

    def __init__(
        self,
        mu,
        sigma,
        lambd,
        obs,
        *,
        extended: ztyping.ExtendedInputType | None = False,
        norm: ztyping.NormInputType | None = None,
        name: str = "Novosibirsk",
        label: str | None = None,
    ):
        """Novosibirsk PDF.

        The Novosibirsk function is a continuous probability density function (PDF) that is used to model
        asymmetric peaks in high-energy physics. It is a theoretical Compton spectrum with a logarithmic Gaussian function.

        .. math::
            f(x;\\sigma, x_0, \\Lambda) = \\exp\\left[
                -\\frac{1}{2} \\frac{\\left( \\ln q_y \\right)^2 }{\\Lambda^2} + \\Lambda^2 \\right] \\\\
            q_y(x;\\sigma,x_0,\\Lambda) = 1 + \\frac{\\Lambda(x-x_0)}{\\sigma} \\times
            \\frac{\\sinh \\left( \\Lambda \\sqrt{\\ln 4} \\right)}{\\Lambda \\sqrt{\\ln 4}}

        Args:
            mu: The peak of the distribution.
            sigma: The width of the distribution.
            lambd: The tail of the distribution.
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
        params = {"mu": mu, "sigma": sigma, "lambd": lambd}
        super().__init__(obs=obs, params=params, name=name, extended=extended, norm=norm, label=label)

    @zfit.supports()
    def _unnormalized_pdf(self, x, params):
        mu = params["mu"]
        sigma = params["sigma"]
        lambd = params["lambd"]
        x = x[0]
        return novosibirsk_pdf(x, mu=mu, sigma=sigma, lambd=lambd)


novosibirsk_integral_limits = Space(axes=0, limits=(ANY_LOWER, ANY_UPPER))
Novosibirsk.register_analytic_integral(func=novosibirsk_integral, limits=novosibirsk_integral_limits)
