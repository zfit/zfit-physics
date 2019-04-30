import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
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
        size = tf.shape(data)[0]
        dims = tf.shape(data)[-1]
        sigma = convert_to_container(sigma)

        # Bandwidth definition, use silverman's rule of thumb for nd
        cov = tf.diag(
            [tf.square((4. / (dims + 2.)) ** (1 / (dims + 4)) * size ** (-1 / (dims + 4)) * s) for s in sigma])
        # kernel prob output shape: (n,)
        kernel = tfd.MultivariateNormalFullCovariance(loc=data, covariance_matrix=cov)
        reshaped_kernel = tfd.Independent(kernel)

        categorical = tfd.Categorical(probs=ztf.constant(1 / size, shape=(size,)))  # no grad -> no need to recreate
        dist_params = dict(mixture_distribution=categorical,
                           components_distribution=reshaped_kernel)
        distribution = tfd.MixtureSameFamily
        super().__init__(distribution=distribution, dist_params=dist_params, obs=obs, name=name)
