import zfit
from zfit.core.space import ANY_LOWER, ANY_UPPER, Space
from zfit import z
import tensorflow as tf
import numpy as np


class RelativisticBreitWigner(zfit.pdf.ZPDF):
    _N_OBS = 1  # dimension, can be omitted
    _PARAMS = ['m', 'Gamma']  # the name of the parameters

    def _unnormalized_pdf(self, x):
        """
        Calculate the PDF at value(s) x.

        Parameters
        ----------
        x : tf.Tensor
            Energies. Either one value or an array
        """
        x = zfit.z.unstack_x(x)
        alpha = self.params['Gamma'] / self.params['m']
        gamma = self.params['m']**2 * (1. + alpha**2)**0.5
        k = 2.**(3. / 2.) * self.params['m']**2 * alpha * gamma / (np.pi * (self.params['m']**2 + gamma)**0.5)

        return k / ((x**2 - self.params['m']**2)**2 + self.params['m']**4 * alpha**2)


def relbw_cdf_func(x, m, Gamma):
    """
    Parameters
    ----------
    x
    m : float
        mass of resonance
    Gamma : float
        width of resonance
    Returns
    -------
    cdf :  float
        CDF of Breit-Wigner distribution
    The CDf was found by Mathematica:
    pdf = k/((m^2 - m^2)^2 + m^4*alpha^2)
    cdf = Integrate[pdf, m]
    >>> BW = breit_wigner(m=125., Gamma=0.05)
    >>> BW.cdf(125.)
    0.50052268648248666
    >>> BW.cdf(1E10)
    1.0000000000000004
    >>> BW.cdf(0.)
    0.0
    """
    alpha = Gamma / m
    gamma = m ** 2 * (1. + alpha ** 2) ** 0.5
    k = 2. ** (3. / 2.) * m ** 2 * alpha * gamma / (np.pi * (m ** 2 + gamma) ** 0.5)

    arg_1 = z.to_complex(-1) ** (1. / 4.) / (-1j + alpha) ** 0.5 * x / m
    arg_2 = z.to_complex(-1) ** (3. / 4.) / (1j + alpha) ** 0.5 * x / m

    shape = -1j * tf.math.atan(arg_1) / (-1j + alpha) ** 0.5 - tf.math.atan(arg_2) / (1j + alpha) ** 0.5
    norm = z.to_complex(-1) ** (1. / 4.) * k / (2. * alpha * m ** 3)

    cdf_ = shape * norm
    cdf_ = z.to_real(cdf_)
    return cdf_


def relbw_integral(limits, params, model):
    lower, upper = limits.rect_limits
    lower_cdf = relbw_cdf_func(x=limits[0], m=params['m'], Gamma=params['Gamma'])
    upper_cdf = relbw_cdf_func(x=limits[1], m=params['m'], Gamma=params['Gamma'])
    return upper - lower


relbw_integral_limits = Space(axes=(0,), limits=(((ANY_LOWER,),), ((ANY_UPPER,),)))
# TODO analytic integral does not work
#RelativisticBreitWigner.register_analytic_integral(func=relbw_integral, limits=relbw_integral_limits)
