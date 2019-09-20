"""Example test for a pdf or function"""
import pytest
import zfit
# Important, do the imports below
from zfit.core.testing import setup_function, teardown_function, tester

import zfit_physics as zphys
import numpy as np

# specify globals here. Do NOT add any TensorFlow but just pure python
param1_true = 0.3
param2_true = 1.2


def test_conv_simple():
    # test special properties  here
    obs = zfit.Space("obs1", limits=(-5, 5))
    param1 = zfit.Parameter('param1', -1)
    param2 = zfit.Parameter('param2', 1)
    gauss1 = zfit.pdf.Gauss(0., param2, obs=obs)
    uniform1 = zfit.pdf.Uniform(param1, param2, obs=obs)
    conv = zphys.unstable.pdf.ConvPDF(func=uniform1,
                                      kernel=gauss1,
                                      limits=obs, obs=obs)

    x = np.linspace(-5, 5, 1000)
    probs = conv.pdf(x=x)
    integral = conv.integrate(limits=obs)
    probs_np = zfit.run(probs)
    assert pytest.approx(1, rel=1e-3) == zfit.run(integral)
    assert len(probs_np) == 1000
    # assert len(conv.get_dependents(only_floating=False)) == 2  # TODO: activate again with fixed params
