"""ARGUS PDF (https://en.wikipedia.org/wiki/ARGUS_distribution)"""
import numpy as np
import tensorflow as tf
import zfit
from zfit import z


# @z.function_tf
def argus_func(m, m0, c, p):
    m_frac = m / m0
    use_m_frac = tf.logical_and(tf.less(m_frac, z.constant(1.)), tf.greater(m, z.constant(0.)))

    def argus_unsafe(m_frac):
        m_factor = 1 - z.pow(m_frac, 2)
        return m * z.pow(m_factor, p) * (z.exp(c * m_factor))

    # if m > m0 (mass larger then the cutoff mass), this blows up. Set zero if m > m0
    argus_filtered = z.safe_where(condition=use_m_frac,
                                  func=argus_unsafe,
                                  safe_func=lambda x: tf.zeros_like(x),
                                  values=m_frac,
                                  value_safer=lambda x: tf.ones_like(x) * 0.5,
                                  )
    return argus_filtered


class Argus(zfit.pdf.ZPDF):
    """Implementation of the ARGUS PDF"""

    _N_OBS = 1
    _PARAMS = "m0", "c", "p"

    def _unnormalized_pdf(self, x):
        """
        Calculation of ARGUS PDF value
        (Docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.argus.html)
        """
        m = zfit.ztf.unstack_x(x)

        m0 = self.params["m0"]
        c = self.params["c"]
        p = self.params["p"]

        return argus_func(m, m0, c, p)


# @z.function_tf
def uppergamma(s, x):
    return tf.math.igammac(s, x=x) * z.exp(tf.math.lgamma(x))


# @z.function_tf
def argus_cdf_p_half(lim, c, m0):
    lim = tf.where(tf.less_equal(lim, m0), lim, m0)  # take the smaller one, only integrate up to m0
    lim = tf.where(tf.greater(lim, 0.), lim, z.constant(0.))  # start from 0 as minimum

    lim_square = z.square(lim)
    m0_squared = z.square(m0)
    return (-0.5 * m0_squared * z.pow(-c * (1 - lim_square / m0_squared), -0.5)
            * z.sqrt(1 - lim_square / m0_squared) * uppergamma((z.constant(1.5)),
                                                               -c * (1 - lim_square / m0_squared)) / c)


def argus_integral_p_half_func(lower, upper, c, m0):
    return argus_cdf_p_half(upper, c=c, m0=m0) - argus_cdf_p_half(lower, c=c, m0=m0)


def argus_integral_p_half(limits, params, model):
    p = params["p"]
    if not isinstance(p, zfit.param.ConstantParameter) or not np.isclose(p.value(), 0.5):
        raise NotImplementedError
    m0 = params["m0"]
    c = params["c"]
    lower, upper = limits.limit1d
    lower = z.convert_to_tensor(lower)
    upper = z.convert_to_tensor(upper)
    integral = argus_integral_p_half_func(lower=lower, upper=upper, c=c, m0=m0)
    return integral


argus_integral_limits = zfit.Space.from_axes(axes=(0,), limits=(((zfit.Space.ANY_LOWER,),), ((zfit.Space.ANY_UPPER,),)))
Argus.register_analytic_integral(func=argus_integral_p_half, limits=argus_integral_limits)

# TODO: works for negative c, but for positive?
# require limits for parameters in pdf?
# TODO: analytic integral is off
if __name__ == '__main__':
    obs = zfit.Space('obs1', (0.3, 4.))
    argus = Argus(m0=5., c=-3., p=0.5, obs=obs)
    print(argus.pdf(tf.linspace(0.1, 15., 100)))
    argus_pdf = argus.pdf(tf.linspace(0.3, 4, 1000001))
    print(tf.reduce_mean(argus_pdf) * obs.area())
    import sympy as sp

    N = sp.Symbol('N')
    m = sp.Symbol('m')
    m0 = sp.Symbol('m0')
    c = sp.Symbol('c')
    t = sp.Symbol('t')
    mu = sp.Symbol('mu')
    sigma = sp.Symbol('sigma')

    # p = sp.Symbol('p')
    p = 0.5
    low = sp.Symbol('low')
    lim = sp.Symbol('up')

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
    global_assumptions.add(sp.Q.integer(p))
    global_assumptions.add(sp.Q.finite(c))
    integral = sp.integrate(
        (N * m * (1 - (m / m0) ** 2) ** p * sp.exp(c * (1 - (m / m0) ** 2))) *
        sp.exp((m - t - mu) ** 2 / sigma ** 2), t)
    print(integral)
    func1 = sp.lambdify(integral.free_symbols, integral, 'tensorflow')
    import inspect

    source = inspect.getsource(func1)
    print(source)
    # sp.lambdify()
