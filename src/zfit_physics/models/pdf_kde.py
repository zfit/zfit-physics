from __future__ import annotations

from collections import OrderedDict

import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
import zfit
from zfit import z
from zfit.models.dist_tfp import WrapDistribution
from zfit.util import ztyping
from zfit.util.container import convert_to_container
from zfit.util.exception import WorkInProgressError


class GaussianKDE(WrapDistribution):  # multidimensional kde with gaussian kernel
    def __init__(
        self,
        data: tf.Tensor,
        bandwidth: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ztyping.ParamTypeInput | None = None,
        norm: ztyping.NormTypeInput = None,
        name: str = "GaussianKDE",
        label: str | None = None,
    ):
        """Gaussian Kernel Density Estimation using Silverman's rule of thumb.

        Args:
            data: Data points to build a kernel around
            bandwidth: sigmas for the covariance matrix of the multivariate gaussian
            obs: |@doc:pdf.init.obs| Observables of the
               model. This will be used as the default space of the PDF and,
               if not given explicitly, as the normalization range.

               The default space is used for example in the sample method: if no
               sampling limits are given, the default space is used.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               The observables are not equal to the domain as it does not restrict or
               truncate the model outside this range. |@docend:pdf.init.obs|
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: |@doc:pdf.init.name| Name of the PDF.
               Maybe has implications on the serialization and deserialization of the PDF.
               For a human-readable name, use the label. |@docend:pdf.init.name|
           label: |@doc:pdf.init.label| Human-readable name
               or label of
               the PDF for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.label|
        """
        dtype = zfit.settings.ztypes.float
        if isinstance(data, zfit.core.interfaces.ZfitData):
            msg = "Currently, no dataset supported yet"
            raise WorkInProgressError(msg)
            # size = data.nevents
            # dims = data.n_obs
            # with data.
            # data = data.value()
            # if data.weights is not None:

        if not isinstance(data, tf.Tensor):
            data = z.convert_to_tensor(value=data)
        data = z.to_real(data)

        shape_data = tf.shape(data)
        size = tf.cast(shape_data[0], dtype=dtype)
        dims = tf.cast(shape_data[-1], dtype=dtype)
        bandwidth = convert_to_container(bandwidth)

        # Bandwidth definition, use silverman's rule of thumb for nd
        def reshaped_kerner_factory():
            cov_diag = [
                tf.square((4.0 / (dims + 2.0)) ** (1 / (dims + 4)) * size ** (-1 / (dims + 4)) * s) for s in bandwidth
            ]
            # cov = tf.linalg.diag(cov_diag)
            # kernel prob output shape: (n,)
            # kernel = tfd.MultivariateNormalFullCovariance(loc=data, covariance_matrix=cov)
            return tfd.MultivariateNormalDiag(loc=data, scale_diag=cov_diag)

            # return tfd.Independent(kernel)

        # reshaped_kernel = kernel

        probs = tf.broadcast_to(1 / size, shape=(tf.cast(size, tf.int32),))
        categorical = tfd.Categorical(probs=probs)  # no grad -> no need to recreate

        def dist_kwargs():
            return {"mixture_distribution": categorical, "components_distribution": reshaped_kerner_factory()}

        distribution = tfd.MixtureSameFamily
        # TODO lambda for params
        params = OrderedDict((f"bandwidth_{i}", h) for i, h in enumerate(bandwidth))
        super().__init__(
            distribution=distribution,
            dist_params={},
            dist_kwargs=dist_kwargs,
            params=params,
            obs=obs,
            name=name,
            extended=extended,
            norm=norm,
            label=label,
        )

    # @zfit.supports()
    # def _analytic_integrate(self, limits, norm):
    #     raise AnalyticIntegralNotImplementedError
