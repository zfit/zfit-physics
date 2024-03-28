"""Example test for a pdf or function."""
import numpy as np
import pytest
import tensorflow as tf
import zfit

import zfit_physics as zphys

# specify globals here. Do NOT add any TensorFlow but just pure python
param1_true = 0.3
param2_true = 1.2


def test_conv_simple():
    # test special properties  here
    n_points = 2000
    obs = zfit.Space("obs1", limits=(-5, 5))
    param1 = zfit.Parameter("param1", -3)
    param2 = zfit.Parameter("param2", 0.3)
    gauss1 = zfit.pdf.Gauss(0.0, param2, obs=obs)
    uniform1 = zfit.pdf.Uniform(param1, param2, obs=obs)
    conv = zphys.unstable.pdf.NumConvPDFUnbinnedV1(func=uniform1, kernel=gauss1, limits=obs, obs=obs)

    x = tf.linspace(-5.0, 5.0, n_points)
    probs = conv.pdf(x=x)
    integral = conv.integrate(limits=obs)
    probs_np = probs.numpy()
    assert pytest.approx(1, rel=1e-3) == integral.numpy()
    assert len(probs_np) == n_points
    # import matplotlib.pyplot as plt
    # plt.plot(x, probs_np)
    # plt.show()
    # assert len(conv.get_dependents(only_floating=False)) == 2  # TODO: activate again with fixed params
