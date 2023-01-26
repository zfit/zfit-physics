import numpy as np
import tensorflow as tf
import zfit
from zfit import z
from zfit.core.space import ANY_LOWER, ANY_UPPER, Space
from zfit.util import ztyping


@z.function(wraps="tensor")
def relbw_pdf_func(x, m, gamma):
    """Calculate the relativistic Breit-Wigner PDF.

    Args:
         x: value(s) for which the CDF will be calculated.
         m: Mean value
         gamma: width

    Returns:
        `tf.Tensor`: The calculated PDF values.

    Notes:
        Based on code from this [github gist](https://gist.github.com/andrewfowlie/cd0ed7e6c96f7c9e88f85eb3b9665b97#file-bw-py-L87-L110)
    """
    x = z.unstack_x(x)
    alpha = gamma / m
    gamma2 = m**2 * (1.0 + alpha**2) ** 0.5
    k = 2.0 ** (3.0 / 2.0) * m**2 * alpha * gamma2 / (np.pi * (m**2 + gamma2) ** 0.5)

    return k / ((x**2 - m**2) ** 2 + m**4 * alpha**2)


class RelativisticBreitWigner(zfit.pdf.ZPDF):
    """Relativistic Breit-Wigner distribution.

    Formula for PDF and CDF are based on https://gist.github.com/andrewfowlie/cd0ed7e6c96f7c9e88f85eb3b9665b97

    Args:
        m: the average value
        gamma: the width of the distribution
    """

    _N_OBS = 1
    _PARAMS = ["m", "gamma"]

    def _unnormalized_pdf(self, x: tf.Tensor) -> tf.Tensor:
        """Calculate the PDF at value(s) x.

        Args:
            x : Either one value or an array

        Returns:
            `tf.Tensor`: The value(s) of the unnormalized PDF at x.
        """
        return relbw_pdf_func(x, m=self.params["m"], gamma=self.params["gamma"])


@z.function(wraps="tensor")
def arctan_complex(x):
    r"""Function that evaluates arctan(x) using tensorflow but also supports complex numbers. It is defined as.

    .. math::

        \mathrm{arctan}(x) = \frac{i}{2} \left(\ln(1-ix) - \ln(1+ix)\right)

    Args:
        x: tf.Tensor

    Returns:
        .. math:: \mathrm{arctan}(x)

    Notes:
        Formula is taken from https://www.wolframalpha.com/input/?i=arctan%28a%2Bb*i%29
    TODO: move somewhere?
    """
    return 1 / 2 * 1j * (tf.math.log(1 - 1j * x) - tf.math.log(1 + 1j * x))


@z.function(wraps="tensor")
def relbw_cdf_func(x, m, gamma):
    """Analytical function for the CDF of the relativistic Breit-Wigner distribution.

    Args:
         x: value(s) for which the CDF will be calculated.
         m: Mean value
         gamma: width

    Returns:
        `tf.Tensor`: The calculated CDF values.

    Notes:
        Based on code from this [github gist](https://gist.github.com/andrewfowlie/cd0ed7e6c96f7c9e88f85eb3b9665b97#file-bw-py-L112-L154)
    """
    gamma = z.to_complex(gamma)
    m = z.to_complex(m)
    x = z.to_complex(z.unstack_x(x))

    alpha = gamma / m
    gamma2 = m**2 * (1.0 + alpha**2) ** 0.5
    k = 2.0 ** (3.0 / 2.0) * m**2 * alpha * gamma2 / (np.pi * (m**2 + gamma2) ** 0.5)

    arg_1 = z.to_complex(-1) ** (1.0 / 4.0) / (-1j + alpha) ** 0.5 * x / m
    arg_2 = z.to_complex(-1) ** (3.0 / 4.0) / (1j + alpha) ** 0.5 * x / m

    shape = -1j * arctan_complex(arg_1) / (-1j + alpha) ** 0.5 - arctan_complex(arg_2) / (1j + alpha) ** 0.5
    norm = z.to_complex(-1) ** (1.0 / 4.0) * k / (2.0 * alpha * m**3)

    cdf_ = shape * norm
    cdf_ = z.to_real(cdf_)
    return cdf_


def relbw_integral(limits: ztyping.SpaceType, params: dict, model) -> tf.Tensor:
    """Calculates the analytic integral of the relativistic Breit-Wigner PDF.

    Args:
        limits: An object with attribute rect_limits.
        params: A hashmap from which the parameters that defines the PDF will be extracted.
        model: Will be ignored.

    Returns:
        The calculated integral.
    """
    lower, upper = limits.rect_limits
    lower_cdf = relbw_cdf_func(x=lower, m=params["m"], gamma=params["gamma"])
    upper_cdf = relbw_cdf_func(x=upper, m=params["m"], gamma=params["gamma"])
    return upper_cdf - lower_cdf


# These lines of code adds the analytic integral function to RelativisticBreitWigner PDF.
relbw_integral_limits = Space(axes=(0,), limits=(((ANY_LOWER,),), ((ANY_UPPER,),)))
RelativisticBreitWigner.register_analytic_integral(func=relbw_integral, limits=relbw_integral_limits)
