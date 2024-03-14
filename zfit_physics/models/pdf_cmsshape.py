from typing import Optional

import numpy as np
import tensorflow as tf
import zfit
from zfit import z
from zfit.core.space import ANY_LOWER, ANY_UPPER, Space
from zfit.util import ztyping


@z.function(wraps="tensor")
def cmsshape_pdf_func(x, beta, gamma, m):
    x = z.unstack_x(x)
    half = 0.5
    two = 2.0
    t1 = tf.math.exp(-gamma * (x - m))
    t2 = tf.math.erfc(-beta * (x - m))
    t3 = half * gamma * tf.math.exp(-((half * gamma / beta) ** two))
    return t1 * t2 * t3


@z.function(wraps="tensor")
def cmsshape_cdf_func(x, beta, gamma, m):
    half = 0.5
    two = 2.0
    y = x - m
    t1 = tf.math.erf(gamma / (two * beta) + beta * y)
    t2 = tf.math.exp(-((gamma / (two * beta)) ** two) - gamma * y)
    t3 = tf.math.erfc(-beta * y)
    return half * (t1 - t2 * t3) + half


def cmsshape_integral(limits: ztyping.SpaceType, params: dict, model) -> tf.Tensor:
    lower, upper = limits.rect_limits
    beta = params["beta"]
    gamma = params["gamma"]
    m = params["m"]
    lower_cdf = cmsshape_cdf_func(x=lower, beta=beta, gamma=gamma, m=m)
    upper_cdf = cmsshape_cdf_func(x=upper, beta=beta, gamma=gamma, m=m)
    return upper_cdf - lower_cdf


class CMSShape(zfit.pdf.BasePDF):
    _N_OBS = 1

    def __init__(
        self,
        beta: ztyping.ParamTypeInput,
        gamma: ztyping.ParamTypeInput,
        m: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: Optional[ztyping.ExtendedInputType] = None,
        norm: Optional[ztyping.NormInputType] = None,
        name: str = "CMSShape",
    ):
        params = {"beta": beta, "gamma": gamma, "m": m}
        super().__init__(obs=obs, params=params, name=name, extended=extended, norm=norm)

    def _unnormalized_pdf(self, x: tf.Tensor) -> tf.Tensor:
        beta = self.params["beta"]
        gamma = self.params["gamma"]
        m = self.params["m"]
        return cmsshape_pdf_func(x, beta, gamma, m)


cmsshape_integral_limits = Space(axes=(0,), limits=(((ANY_LOWER,),), ((ANY_UPPER,),)))
CMSShape.register_analytic_integral(func=cmsshape_integral, limits=cmsshape_integral_limits)
