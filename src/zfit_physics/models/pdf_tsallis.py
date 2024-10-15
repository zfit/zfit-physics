from __future__ import annotations

import tensorflow as tf
import zfit
import zfit.z.numpy as znp
from zfit import run, z
from zfit.core.space import ANY_LOWER, ANY_UPPER, Space
from zfit.util import ztyping


@z.function(wraps="tensor")
def tsallis_pdf_func(x, m, t, n):
    """Calculate the Tsallis PDF.

    Args:
        x: value(s) for which the PDF will be calculated.
        m: mass of the particle.
        t: width parameter.
        n: absolute value of the exponent of the power law.

    Returns:
        `tf.Tensor`: The calculated PDF values.

    Notes:
        Based on code from `numba-stats <https://github.com/HDembinski/numba-stats/blob/main/src/numba_stats/tsallis.py>`_.
        Formula from CMS, Eur. Phys. J. C (2012) 72:2164
    """
    if run.executing_eagerly():
        if n <= 2:
            msg = "n > 2 is required"
            raise ValueError(msg)
    elif run.numeric_checks:
        tf.debugging.assert_greater(n, znp.array(2.0), message="n > 2 is required")

    mt = znp.hypot(m, x)
    nt = n * t
    c = (n - 1) * (n - 2) / (nt * (nt + (n - 2) * m))
    return c * x * znp.power(1 + (mt - m) / nt, -n)


@z.function(wraps="tensor")
def tsallis_cdf_func(x, m, t, n):
    """Calculate the Tsallis CDF.

    Args:
        x: value(s) for which the CDF will be calculated.
        m: mass of the particle.
        t: width parameter.
        n: absolute value of the exponent of the power law.

    Returns:
        `tf.Tensor`: The calculated CDF values.

    Notes:
        Based on code from `numba-stats <https://github.com/HDembinski/numba-stats/blob/main/src/numba_stats/tsallis.py>`_.
        Formula from CMS, Eur. Phys. J. C (2012) 72:2164
    """
    if run.executing_eagerly():
        if n <= 2:
            msg = "n > 2 is required"
            raise ValueError(msg)
    elif run.numeric_checks:
        tf.debugging.assert_greater(n, znp.asarray(2.0), "n > 2 is required")

    x = z.unstack_x(x)
    mt = znp.hypot(m, x)
    nt = n * t
    return znp.power((mt - m) / nt + 1, 1 - n) * (m + mt - n * (mt + t)) / (m * (n - 2) + nt)


def tsallis_integral(limits: ztyping.SpaceType, params: dict, model) -> tf.Tensor:
    """Calculates the analytic integral of the Tsallis PDF.

    Args:
        limits: An object with attribute limit1d.
        params: A hashmap from which the parameters that defines the PDF will be extracted.
        model: Will be ignored.

    Returns:
        The calculated integral.
    """
    del model
    lower, upper = limits._rect_limits_tf
    m = params["m"]
    t = params["t"]
    n = params["n"]
    lower_cdf = tsallis_cdf_func(x=lower, m=m, t=t, n=n)
    upper_cdf = tsallis_cdf_func(x=upper, m=m, t=t, n=n)
    return upper_cdf - lower_cdf


class Tsallis(zfit.pdf.BasePDF):
    _N_OBS = 1

    def __init__(
        self,
        m: ztyping.ParamTypeInput,
        t: ztyping.ParamTypeInput,
        n: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ztyping.ExtendedInputType | None = None,
        norm: ztyping.NormInputType | None = None,
        name: str = "Tsallis",
        label: str | None = None,
    ):
        """Tsallis-Hagedorn PDF.

        A generalisation (q-analog) of the exponential distribution based on Tsallis entropy.
        It approximately describes the pT distribution charged particles produced in high-energy
        minimum bias particle collisions.


        Based on code from `numba-stats <https://github.com/HDembinski/numba-stats/blob/main/src/numba_stats/tsallis.py>`_.
        Formula from CMS, Eur. Phys. J. C (2012) 72:2164

        Args:
            m: Mass of the particle.
            t: Width parameter.
            n: Absolute value of the exponent of the power law.
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
        if run.executing_eagerly():
            if n <= 2:
                msg = "n > 2 is required"
                raise ValueError(msg)
        elif run.numeric_checks:
            tf.debugging.assert_greater(n, znp.asarray(2.0), message="n > 2 is required")

        params = {"m": m, "t": t, "n": n}
        super().__init__(obs=obs, params=params, name=name, extended=extended, norm=norm, label=label)

    @zfit.supports()
    def _unnormalized_pdf(self, x: tf.Tensor, params) -> tf.Tensor:
        m = params["m"]
        t = params["t"]
        n = params["n"]
        x = x[0]
        return tsallis_pdf_func(x=x, m=m, t=t, n=n)


tsallis_integral_limits = Space(axes=0, limits=(ANY_LOWER, ANY_UPPER))
Tsallis.register_analytic_integral(func=tsallis_integral, limits=tsallis_integral_limits)
