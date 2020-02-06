"""ARGUS PDF (https://en.wikipedia.org/wiki/ARGUS_distribution)"""
import zfit


class Argus(zfit.pdf.ZPDF):
    """Implementation of the ARGUS PDF"""

    _N_OBS = 1
    _PARAMS = "m0 c p".split()

    def _unnormalized_pdf(self, x):
        """
        Calculation of ARGUS PDF value
        (Docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.argus.html)
        """
        m = zfit.ztf.unstack_x(x)

        m0 = self.params["m0"]
        c = self.params["c"]
        p = self.params["p"]

        m_frac = m / m0
        m_factor = 1 - zfit.ztf.pow(m_frac, 2)
        return m * (m_factor ** p) * (zfit.ztf.exp(c * m_factor))
