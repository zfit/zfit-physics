import tensorflow as tf
import tensorflow_probability.python.distributions as tfd


class KDE:  # multidimensional kde with gaussian kernel
    def __init__(self, data, sigma):
        n = data.shape[0].value  # ndata
        d = data.shape[1].value  # ndim
        # Bandwidth definition, use silverman's rule of thumb for 2d
        cov = tf.diag([tf.square((4. / (d + 2.)) ** (1 / (d + 4)) * n ** (-1 / (d + 4)) * s) for s in sigma])
        # kernel prob output shape: (n,)
        kern = tfd.Independent(tfd.MultivariateNormalFullCovariance(loc=data, covariance_matrix=cov))
        # sum over n
        self.kde = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=[1 / n] * n),
                                         components_distribution=kern)

    def prob(self, x): return self.kde.prob(x)
