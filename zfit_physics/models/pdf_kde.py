import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
import zfit
from zfit import ztf
from zfit.models.dist_tfp import WrapDistribution
from zfit.util import ztyping
from zfit.util.container import convert_to_container


class GaussianKDE(WrapDistribution):  # multidimensional kde with gaussian kernel
    def __init__(self, data: tf.Tensor, sigma: ztyping.ParamTypeInput, obs: ztyping.ObsTypeInput,
                 name: str = "GaussianKDE"):
        """Gaussian Kernel Density Estimation using Silverman's rule of thumb

        Args:
            data: Data points to build a kernel around
            sigma: sigmas for the covariance matrix of the multivariate gaussian
            obs:
            name: Name of the PDF
        """
        dtype = zfit.settings.ztypes.float
        data = ztf.convert_to_tensor(value=data)
        data = ztf.to_real(data)

        shape_data = tf.shape(data)
        size = tf.cast(shape_data[0], dtype=dtype)
        dims = tf.cast(shape_data[-1], dtype=dtype)
        sigma = convert_to_container(sigma)

        # Bandwidth definition, use silverman's rule of thumb for nd
        cov = tf.diag(
            [tf.square((4. / (dims + 2.)) ** (1 / (dims + 4)) * size ** (-1 / (dims + 4)) * s) for s in sigma])
        # kernel prob output shape: (n,)
        kernel = tfd.MultivariateNormalFullCovariance(loc=data, covariance_matrix=cov)
        reshaped_kernel = tfd.Independent(kernel)
        # reshaped_kernel = kernel

        probs = tf.broadcast_to(1 / size, shape=(tf.cast(size, tf.int32),))
        categorical = tfd.Categorical(probs=probs)  # no grad -> no need to recreate
        dist_kwargs = dict(mixture_distribution=categorical,
                           components_distribution=reshaped_kernel)
        distribution = tfd.MixtureSameFamily
        # TODO lambda for params
        super().__init__(distribution=distribution, dist_params={}, dist_kwargs=dist_kwargs,
                         obs=obs, name=name)
