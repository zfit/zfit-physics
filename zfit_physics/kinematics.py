import tensorflow as tf
import zfit
from zfit import ztf


def metric_tensor():
    """
    Metric tensor for Lorentz space (constant)
    """
    return ztf.constant([-1., -1., -1., 1.], dtype=zfit.settings.ztypes.float)


def mass_squared(vector):
    """Calculate the squared mass for a Lorentz 4-momentum."""
    return ztf.reduce_sum(vector * vector * metric_tensor(), axis=1)


def lorentz_vector(space, time):
    """
    Make a Lorentz vector from spatial and time components
        space : 3-vector of spatial components
        time  : time component
    """
    return tf.concat([space, tf.stack([time], axis=1)], axis=1)

def vector(x, y, z):
    """
    Make a 3-vector from components
    x, y, z : vector components
    """
    return tf.stack([x, y, z], axis=1)
