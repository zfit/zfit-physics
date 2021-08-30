import zfit
from zfit.core.space import ANY_LOWER, ANY_UPPER, Space
from zfit import z
import tensorflow as tf
import numpy as np


def arctan_complex(x):
    """
    Function that evaluates arctan(x) using tensorflow but also supports complex numbers.

    Args: x

    Returns: arctan(x)

    Notes
    -----
    Formula used: https://www.wolframalpha.com/input/?i=arctan%28a%2Bb*i%29
    """
    return 1/2 * 1j * tf.math.log(1 - 1j*x) - 1/2 * 1j * tf.math.log(1 + 1j*x)


class RelativisticBreitWigner(zfit.pdf.ZPDF):
    """
    Relativistic Breit-Wigner distribution.
    Formula for PDF and CDF are based on https://gist.github.com/andrewfowlie/cd0ed7e6c96f7c9e88f85eb3b9665b97
    """
    _N_OBS = 1
    _PARAMS = ['m', 'Gamma']

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
    Gamma = z.to_complex(Gamma)
    m = z.to_complex(m)
    x = z.to_complex(x)

    alpha = Gamma / m
    gamma = m ** 2 * (1. + alpha ** 2) ** 0.5
    k = 2. ** (3. / 2.) * m ** 2 * alpha * gamma / (np.pi * (m ** 2 + gamma) ** 0.5)

    arg_1 = z.to_complex(-1) ** (1. / 4.) / (-1j + alpha) ** 0.5 * x / m
    arg_2 = z.to_complex(-1) ** (3. / 4.) / (1j + alpha) ** 0.5 * x / m

    shape = -1j * arctan_complex(arg_1) / (-1j + alpha) ** 0.5 - arctan_complex(arg_2) / (1j + alpha) ** 0.5
    norm = z.to_complex(-1) ** (1. / 4.) * k / (2. * alpha * m ** 3)

    cdf_ = shape * norm
    cdf_ = z.to_real(cdf_)
    return cdf_


def relbw_integral(limits, params, model):
    lower, upper = limits.rect_limits
    lower_cdf = relbw_cdf_func(x=lower, m=params['m'], Gamma=params['Gamma'])
    upper_cdf = relbw_cdf_func(x=upper, m=params['m'], Gamma=params['Gamma'])
    return upper_cdf - lower_cdf


relbw_integral_limits = Space(axes=(0,), limits=(((ANY_LOWER,),), ((ANY_UPPER,),)))
# TODO analytic integral does not work
RelativisticBreitWigner.register_analytic_integral(func=relbw_integral, limits=relbw_integral_limits)
