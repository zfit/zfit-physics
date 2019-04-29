from functools import reduce

import tensorflow as tf
import zfit
from zfit.util import ztyping
from zfit.util import exception
from zfit.util.container import convert_to_container
from zfit.util.exception import DueToLazynessNotImplementedError


class ConvContrib(zfit.pdf.BasePDF):
    def __init__(self, func, kernel, limits: zfit.Space, obs: ztyping.ObsTypeInput, ndraws, name):
        super().__init__(obs=obs, params={}, name=name)
        if self.space.n_limits == 0:
            raise exception.LimitsNotSpecifiedError("obs have to have limits to define where to integrate over.")
        ndraws = convert_to_container(ndraws)
        if len(ndraws) == 1 and limits.n_obs == 2:
            ndraws = (ndraws[0], ndraws[0])
        if limits.n_obs == 2:
            lower1, lower2, upper1, upper2 = limits.limit2d
            x, y = tf.meshgrid(tf.linspace(lower1, upper1, ndraws[0]),
                               tf.linspace(lower2, upper2, ndraws[1]))
            grid_points = tf.concat([tf.reshape(x, [-1, 1]), tf.reshape(y, [-1, 1])], axis=1)
        elif limits.n_obs == 1:
            lower, upper = limits.limit1d
            grid_points = tf.linspace(lower, upper, ndraws[0])
        else:
            raise DueToLazynessNotImplementedError("Only 1 and 2 dimensional convolution implemented currently.")

        self._grid_points = grid_points  # true vars
        self._kernel_func = kernel  # callable func of reco - true vars
        self._func_values = func(grid_points)  # func of true vars
        self._conv_limits = limits

    def _unnormalized_pdf(self, x):
        x = x.unstack_x()
        TODO continue here
        area = self._conv_limits.area()
        return tf.map_fn(lambda xi: area * tf.reduce_mean(self._func_values * self._kernel_func(xi - self._grid_points)), x)  # func of reco vars

