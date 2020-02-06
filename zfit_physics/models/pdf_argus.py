"""ARGUS PDF (https://en.wikipedia.org/wiki/ARGUS_distribution)"""
import tensorflow as tf
import zfit
from zfit import z


@z.function_tf
def argus_func(m, m0, c, p):
    m_frac = m / m0
    use_m_frac = tf.less(m_frac, z.constant(1.))

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
