from __future__ import annotations

import tensorflow as tf
import zfit
from zfit import z
from zfit.core.space import ANY_LOWER, ANY_UPPER, Space
from zfit.util import ztyping


@z.function(wraps="tensor")
def cmsshape_pdf_func(x, m, beta, gamma):
    """Calculate the CMSShape PDF.

    Args:
        x: value(s) for which the PDF will be calculated.
        m: approximate center of the disribution.
        beta: steepness of the error function.
        gamma: steepness of the exponential distribution.

    Returns:
        `tf.Tensor`: The calculated PDF values.

    Notes:
        Based on code from `spark_tnp <https://gitlab.cern.ch/cms-muonPOG/spark_tnp/-/blob/Spark3/RooCMSShape.cc>`_ and
        `numba-stats <https://github.com/HDembinski/numba-stats/blob/main/src/numba_stats/cmsshape.py>`_.
    """
    half = 0.5
    two = 2.0
    t1 = tf.math.exp(-gamma * (x - m))
    t2 = tf.math.erfc(-beta * (x - m))
    t3 = half * gamma * tf.math.exp(-((half * gamma / beta) ** two))
    return t1 * t2 * t3


@z.function(wraps="tensor")
def cmsshape_cdf_func(x, m, beta, gamma):
    """Analtical function for the CDF of the CMSShape distribution.

    Args:
        x: value(s) for which the CDF will be calculated.
        m: approximate center of the distribution.
        beta: steepness of the error function.
        gamma: steepness of the exponential distribution.

    Returns:
        `tf.Tensor`: The calculated CDF values.

    Notes:
        Based on code from `spark_tnp <https://gitlab.cern.ch/cms-muonPOG/spark_tnp/-/blob/Spark3/RooCMSShape.cc>`_ and
        `numba-stats <https://github.com/HDembinski/numba-stats/blob/main/src/numba_stats/cmsshape.py>`_
    """
    half = 0.5
    two = 2.0
    y = x - m
    t1 = tf.math.erf(gamma / (two * beta) + beta * y)
    t2 = tf.math.exp(-((gamma / (two * beta)) ** two) - gamma * y)
    t3 = tf.math.erfc(-beta * y)
    return half * (t1 - t2 * t3) + half


def cmsshape_integral(limits: ztyping.SpaceType, params: dict, model) -> tf.Tensor:
    """Calculates the analytic integral of the CMSShape PDF.

    Args:
        limits: An object with attribute limit1d.
        params: A hashmap from which the parameters that defines the PDF will be extracted.
        model: Will be ignored.

    Returns:
        The calculated integral.
    """
    del model
    lower, upper = limits.limit1d
    m = params["m"]
    beta = params["beta"]
    gamma = params["gamma"]
    lower_cdf = cmsshape_cdf_func(x=lower, m=m, beta=beta, gamma=gamma)
    upper_cdf = cmsshape_cdf_func(x=upper, m=m, beta=beta, gamma=gamma)
    return upper_cdf - lower_cdf


class CMSShape(zfit.pdf.BasePDF):
    _N_OBS = 1

    def __init__(
        self,
        m: ztyping.ParamTypeInput,
        beta: ztyping.ParamTypeInput,
        gamma: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ztyping.ExtendedInputType | None = None,
        norm: ztyping.NormInputType | None = None,
        name: str = "CMSShape",
        label: str | None = None,
    ):
        """CMSShape PDF.

        The distribution consists of an exponential decay suppressed at small values by the
        complementary error function. The product is an asymmetric peak with a bell shape on the
        left-hand side at low mass due to threshold effect and an exponential tail on the right-hand side.
        This shape is used by the CMS experiment to model the background in the invariant mass distribution
        of Z to ll decay candidates.

        Formula for the PDF and CDF are based on code from
        `spark_tnp <https://gitlab.cern.ch/cms-muonPOG/spark_tnp/-/blob/Spark3/RooCMSShape.cc>`_ and
        `numba-stats <https://github.com/HDembinski/numba-stats/blob/main/src/numba_stats/cmsshape.py>`_

        Args:
            m: Approximate center of the distribution.
            beta: Steepness of the error function.
            gamma: Steepness of the exponential distribution.
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
        params = {"m": m, "beta": beta, "gamma": gamma}
        super().__init__(obs=obs, params=params, name=name, extended=extended, norm=norm, label=label)

    @zfit.supports()
    def _unnormalized_pdf(self, x: tf.Tensor, params) -> tf.Tensor:
        m = params["m"]
        beta = params["beta"]
        gamma = params["gamma"]
        x = x[0]
        return cmsshape_pdf_func(x=x, m=m, beta=beta, gamma=gamma)


cmsshape_integral_limits = Space(axes=(0,), limits=(((ANY_LOWER,),), ((ANY_UPPER,),)))
CMSShape.register_analytic_integral(func=cmsshape_integral, limits=cmsshape_integral_limits)
