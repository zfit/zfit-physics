from functools import reduce
from typing import Callable, Union

import tensorflow as tf
import tensorflow_probability as tfp
import zfit
from zfit import ztf
from zfit.util import ztyping
from zfit.util import exception
from zfit.util.exception import DueToLazynessNotImplementedError

import zfit.models.functor


class ConvPDF(zfit.models.functor.BaseFunctor):
    def __init__(self, func: zfit.pdf.BasePDF, kernel: zfit.pdf.BasePDF,
                 limits: ztyping.ObsTypeInput, obs: ztyping.ObsTypeInput,
                 ndraws: int = 20000, name: str = "Convolution"):
        """Numerical Convolution pdf of *func* convoluted with *kernel*.

        Args:
            func (:py:class:`zfit.pdf.BasePDF`): PDF  with `pdf` method that takes x and returns the function value.
                Here x is a `Data` with the obs and limits of *limits*.
            kernel (:py:class:`zfit.pdf.BasePDF`): PDF with `pdf` method that takes x acting as the kernel.
                Here x is a `Data` with the obs and limits of *limits*.
            limits (:py:class:`zfit.Space`): Limits for the numerical integration.
            obs (:py:class:`zfit.Space`): Observables of the class
            ndraws (int): Number of draws for the mc integration
            name (str): Human readable name of the pdf
        """
        super().__init__(obs=obs, pdfs=[func, kernel], params={}, name=name)
        limits = self._check_input_limits(limits=limits)
        if limits.n_limits == 0:
            raise exception.LimitsNotSpecifiedError("obs have to have limits to define where to integrate over.")
        if limits.n_limits > 1:
            raise DueToLazynessNotImplementedError("Multiple Limits not implemented")

#        if not isinstance(func, zfit.pdf.BasePDF):
#            raise TypeError(f"func has to be a PDF, not {type(func)}")
#        if isinstance(kernel, zfit.pdf.BasePDF):
#            raise TypeError(f"kernel has to be a PDF, not {type(kernel)}")

        func = lambda x: func.unnormalized_pdf(x=x)
        kernel = lambda x: kernel.unnormalized_pdf(x=x)

        self._grid_points = None  # true vars  # callable func of reco - true vars
        self._func_values = None
        self.conv_limits = limits
        self._ndraws = ndraws

    def _unnormalized_pdf(self, x):
        limits = self.conv_limits
        area = limits.area()

        samples = self._grid_points
        func_values = self._func_values
        if func_values is None:
            # create sample for numerical integral
            lower = limits.lower[0]
            upper = limits.upper[0]
            lower = ztf.convert_to_tensor(lower, dtype=self.dtype)
            upper = ztf.convert_to_tensor(upper, dtype=self.dtype)
            samples_normed = tfp.mcmc.sample_halton_sequence(dim=limits.n_obs, num_results=self._ndraws, dtype=self.dtype,
                                                             randomized=False)
            samples = samples_normed * (upper - lower) + lower  # samples is [0, 1], stretch it
            samples = zfit.Data.from_tensor(obs=limits, tensor=samples)
            self._grid_points = samples

            func_values = self.pdfs[0].unnormalized_pdf(samples)  # func of true vars
            self._func_values = func_values

        return tf.map_fn(
            lambda xi: area * tf.reduce_mean(func_values * self.pdfs[1].unnormalized_pdf(xi - samples)),
            x.value())  # func of reco vars
