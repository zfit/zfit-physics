from functools import reduce

import tensorflow as tf
import tensorflow_probability as tfp
import zfit
from zfit import ztf
from zfit.util import ztyping
from zfit.util import exception
from zfit.util.container import convert_to_container
from zfit.util.exception import DueToLazynessNotImplementedError


class ConvPDF(zfit.pdf.BasePDF):
    def __init__(self, func, kernel, obs: ztyping.ObsTypeInput, ndraws=20000, name="Convolution"):
        super().__init__(obs=obs, params={}, name=name)
        limits = self.space
        if limits.n_limits == 0:
            raise exception.LimitsNotSpecifiedError("obs have to have limits to define where to integrate over.")
        if limits.n_limits > 1:
            raise DueToLazynessNotImplementedError("Multiple Limits not implemented")

        if isinstance(func, zfit.pdf.BasePDF):
            func = lambda x: func.unnormalized_pdf(x=x)
        if isinstance(kernel, zfit.pdf.BasePDF):
            kernel = lambda x: kernel.unnormalized_pdf(x=x)

        lower = limits.lower[0]
        upper = limits.upper[0]
        lower = ztf.convert_to_tensor(lower, dtype=self.dtype)
        upper = ztf.convert_to_tensor(upper, dtype=self.dtype)
        samples_normed = tfp.mcmc.sample_halton_sequence(dim=limits.n_obs, num_results=ndraws, dtype=self.dtype,
                                                         randomized=False)
        samples = samples_normed * (upper - lower) + lower  # samples is [0, 1], stretch it
        sample_data = zfit.Data.from_tensor(obs=limits, tensor=samples)

        self._grid_points = samples  # true vars
        self._kernel_func = kernel  # callable func of reco - true vars
        self._func_values = func(sample_data)  # func of true vars
        self._conv_limits = limits

    def _unnormalized_pdf(self, x):
        area = self._conv_limits.area()
        return tf.map_fn(
            lambda xi: area * tf.reduce_mean(self._func_values * self._kernel_func(xi - self._grid_points)),
            x)  # func of reco vars
