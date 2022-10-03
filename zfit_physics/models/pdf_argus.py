"""ARGUS PDF (https://en.wikipedia.org/wiki/ARGUS_distribution)"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import zfit
from zfit import z
from zfit.util import ztyping


@z.function(wraps="tensor")
def argus_func(
    m: ztyping.NumericalType,
    m0: ztyping.NumericalType,
    c: ztyping.NumericalType,
    p: ztyping.NumericalType,
) -> tf.Tensor:
    r"""`ARGUS shape <https://en.wikipedia.org/wiki/ARGUS_distribution>`_ describing the invariant mass of a particle in
    a continuous background.

    It is defined as

    .. math::

        \mathrm{Argus}(m, m_0, c, p) = m \cdot \left[ 1 - \left( \frac{m}{m_0} \right)^2 \right]^p
        \cdot \exp\left[ c \cdot \left(1 - \left(\frac{m}{m_0}\right)^2 \right) \right]

    The implementation follows the `RooFit version <https://root.cern.ch/doc/master/classRooArgusBG.html>`_

    Args:
        m: Mass of the particle
        m0: Maximal energetically allowed mass, cutoff
        c: peakiness of the distribution
        p: Generalized ARGUS shape, for p = 0.5, the normal ARGUS shape is recovered

    Returns:
        `tf.Tensor`: the values matching the (broadcasted) shapes of the input
    """
    m = tfp.math.clip_by_value_preserve_gradient(m, 0.0, m0)
    m_frac = m / m0

    m_factor = 1 - z.square(m_frac)
    argus = m * z.pow(m_factor, p) * (z.exp(c * m_factor))
    return argus


class Argus(zfit.pdf.BasePDF):
    def __init__(self, obs: ztyping.ObsTypeInput, m0, c, p, name: str = "ArgusPDF"):
        r"""`ARGUS shape <https://en.wikipedia.org/wiki/ARGUS_distribution>`_ describing the invariant mass of a particle
        in a continuous background.

        The ARGUS shaped function describes the reconstructed invariant mass of a decayed particle, especially at the
        kinematic boundaries of the maximum beam energy. It is defined as

        .. math::

            \mathrm{Argus}(m, m_0, c, p) = m \cdot \left[ 1 - \left( \frac{m}{m_0} \right)^2 \right]^p
            \cdot \exp\left[ c \cdot \left(1 - \left(\frac{m}{m_0}\right)^2 \right) \right]

        and normalized to one over the `norm_range` (which defaults to `obs`).

        The implementation follows the `RooFit version <https://root.cern.ch/doc/master/classRooArgusBG.html>`_

        Args:
            m0: Maximal energetically allowed mass, cutoff
            c: Shape parameter; "peakiness" of the distribution
            p: Generalization of the ARGUS shape, for p = 0.5, the normal ARGUS shape is recovered

        Returns:
            `tf.Tensor`: the values matching the (broadcasted) shapes of the input
        """
        params = {"m0": m0, "c": c, "p": p}
        super().__init__(obs=obs, name=name, params=params)

    _N_OBS = 1

    def _unnormalized_pdf(self, x):
        """
        Calculation of ARGUS PDF value
        (Docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.argus.html)
        """
        m = zfit.z.unstack_x(x)

        m0 = self.params["m0"]
        c = self.params["c"]
        p = self.params["p"]
        return argus_func(m, m0, c, p)


# Keep? move to math?
# @z.function_tf
def uppergamma(s, x):
    return tf.math.igammac(s, x=x) * z.exp(tf.math.lgamma(x))


@z.function(wraps="tensor")
def argus_cdf_p_half_nonpositive(lim, c, m0):
    lim = tf.clip_by_value(lim, 0.0, m0)
    cdf = tf.cond(
        tf.math.less(c, 0.0),
        lambda: argus_cdf_p_half_c_neg(lim=lim, c=c, m0=m0),
        lambda: argus_cdf_p_half_c_zero(lim=lim, c=c, m0=m0),
    )
    return cdf


# Does not work, why?
# # @z.function_tf
# def argus_cdf_p_half_sympy(lim, c, m0):
#     # lim = tf.where(tf.less_equal(lim, m0), lim, m0)  # take the smaller one, only integrate up to m0
#     # lim = tf.where(tf.greater(lim, 0.), lim, z.constant(0.))  # start from 0 as minimum
#     lim = tf.clip_by_value(lim, 0., m0)
#     lim_square = z.square(lim)
#     m0_squared = z.square(m0)
#     return (-0.5 * m0_squared * z.pow(-c * (1 - lim_square / m0_squared), -0.5)
#             * z.sqrt(1 - lim_square / m0_squared) * uppergamma((z.constant(1.5)),
#                                                                -c * (1 - lim_square / m0_squared)) / c)


@z.function(wraps="tensor")
def argus_cdf_p_half_c_neg(lim, c, m0):
    f1 = 1 - z.square(lim / m0)
    cdf = -0.5 * z.square(m0)
    cdf *= z.exp(c * f1) * z.sqrt(f1) / c + 0.5 / z.pow(-c, 1.5) * z.sqrt(z.pi) * tf.math.erf(z.sqrt(-c * f1))
    return cdf


@z.function(wraps="tensor")
def argus_cdf_p_half_c_zero(lim, c, m0):
    del c
    f1 = 1 - z.square(lim / m0)
    cdf = -z.square(m0) / 3.0 * f1 * z.sqrt(f1)
    return cdf


# TODO: add Faddeev function approximation
# def argus_cdf_p_half_c_pos(lim, c, m0):
#     f1 = 1 - z.square(lim)
#     cdf = 0.5 * z.square(m0) * z.exp(c * f1) / (c * z.sqrt(c))
#     # cdf *= (0.5 * z.sqrt(z.pi) * (RooMath::faddeeva(sqrt(c * f1))).imag() - z.sqrt(c * f1))
#     return cdf


@z.function(wraps="tensor")
def argus_integral_p_half_func(lower, upper, c, m0):
    return argus_cdf_p_half_nonpositive(upper, c=c, m0=m0) - argus_cdf_p_half_nonpositive(lower, c=c, m0=m0)


def argus_integral_p_half(limits, params, model):
    p = params["p"]
    if not isinstance(p, zfit.param.ConstantParameter) or not np.isclose(p.static_value, 0.5):
        raise zfit.exception.AnalyticIntegralNotImplementedError()
    c = params["c"]
    if not isinstance(c, zfit.param.ConstantParameter) or c.static_value > 0:
        raise zfit.exception.AnalyticIntegralNotImplementedError()

    m0 = params["m0"]
    lower, upper = limits.limit1d
    lower = z.convert_to_tensor(lower)
    upper = z.convert_to_tensor(upper)
    integral = argus_integral_p_half_func(lower=lower, upper=upper, c=c, m0=m0)
    return integral


argus_integral_limits = zfit.Space(axes=(0,), limits=(zfit.Space.ANY_LOWER, zfit.Space.ANY_UPPER))
Argus.register_analytic_integral(func=argus_integral_p_half, limits=argus_integral_limits)

if __name__ == "__main__":
    # create the integral
    import sympy as sp

    N = sp.Symbol("N")
    m = sp.Symbol("m")
    m0 = sp.Symbol("m0")
    c = sp.Symbol("c")
    t = sp.Symbol("t")
    mu = sp.Symbol("mu")
    sigma = sp.Symbol("sigma")

    # p = sp.Symbol('p')
    p = 0.5
    low = sp.Symbol("low")
    lim = sp.Symbol("up")

    from sympy.assumptions.assume import global_assumptions

    global_assumptions.add(sp.Q.positive(N))
    global_assumptions.add(sp.Q.finite(N))
    global_assumptions.add(sp.Q.positive(sigma))
    global_assumptions.add(sp.Q.finite(sigma))
    global_assumptions.add(sp.Q.positive(m))
    global_assumptions.add(sp.Q.finite(m))
    global_assumptions.add(sp.Q.positive(m / m0))
    global_assumptions.add(sp.Q.finite(m / m0))
    global_assumptions.add(sp.Q.positive(p))
    global_assumptions.add(sp.Q.finite(p))
    # global_assumptions.add(sp.Q.integer(p))
    global_assumptions.add(sp.Q.finite(c))
    global_assumptions.add(sp.Q.positive(c))
    m_factor = 1 - (m / m0) ** 2
    integral_expression = m * m_factor**p * (sp.exp(c * m_factor))
    # integral_expression = (N * m * (1 - (m / m0) ** 2) ** p * sp.exp(c * (1 - (m / m0) ** 2)))
    integral = sp.integrate(integral_expression, m)
    print(integral)
    func1 = sp.lambdify(integral.free_symbols, integral, "tensorflow")
    import inspect

    source = inspect.getsource(func1)
    print(source)
    # sp.lambdify()
