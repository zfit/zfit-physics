import tensorflow as tf
import zfit
from zfit import ztf
from zfit.util import ztyping

from .. import kinematics


def relativistic_breit_wigner(m2, mres, wres):
    """Relativistic Breit-Wigner function.
    """
    second_part = tf.complex(ztf.constant(0.), mres) * ztf.to_complex(wres)
    below_div = ztf.to_complex(mres ** 2 - m2) - second_part
    return 1. / below_div


class RelativisticBreitWigner(zfit.func.BaseFunc):

    def __init__(self, obs: ztyping.ObsTypeInput, mres: float, wres: float, using_m_squared: bool = False,
                 name: str = "RelativisticBreitWigner"):
        """Relativistic Breit Wigner Function

        Args:
            obs: Space the function is defined in
            mres: Mass of the resonance
            wres: width of the resonance mass
            using_m_squared: Whether the input mass is already the squared mass. If not, will be squared on the
                fly. If the input is a list, the kinematics of daughter particles is assumed and the mass squared is
                calculated
            name: Name of the Function
        """
        self.using_m_squared = using_m_squared
        # HACK to make it usable in while loop
        # zfit.run._enable_parameter_autoconversion = False

        super().__init__(obs=obs, name=name, dtype=zfit.settings.ztypes.complex,
                         params={'mres': mres, 'wres': wres})

        # zfit.run._enable_parameter_autoconversion = True
        # HACK end

    def _func(self, x):
        var = x.unstack_x()
        if isinstance(var, list):
            m_sq = kinematics.mass_squared(tf.reduce_sum(
                [kinematics.lorentz_vector(kinematics.vector(px, py, pz), pe)
                 for px, py, pz, pe in zip(*[iter(var)] * 4)],
                axis=0))
        elif self.using_m_squared:
            m_sq = var
        else:
            m_sq = var * tf.math.conj(var)  # TODO(Albert): this was squared, but should be mult with conj, right?
        mres = self.params['mres']
        wres = self.params['wres']
        return relativistic_breit_wigner(m_sq, mres, wres)
