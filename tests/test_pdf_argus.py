"""Example test for a pdf or function"""

import numpy as np
import pytest
import tensorflow as tf
import zfit
# Important, do the imports below
from zfit.core.testing import tester

# specify globals here. Do NOT add any TensorFlow but just pure python
param1_true = 0.3
param2_true = 1.2


def test_standard():
    # test special properties  here
    obs = zfit.Space('obs1', (-2, 6))
    from zfit_physics.models.tmp_pdf_argus import ArgusPDF
    argus = ArgusPDF(m0=5., c=-3., p=0.5, obs=obs)
    assert not any(np.isnan(argus.pdf(tf.linspace(0.1, 15., 100))))
    lower = 0.
    upper = 5.
    argus_pdf = argus.pdf(tf.linspace(lower, upper, 1000001))
    assert pytest.approx(zfit.run(tf.reduce_mean(argus_pdf) * (upper - lower)), 4e-2) == 1.
    analytic_integral = zfit.run(argus.analytic_integrate(obs, norm_range=False))
    numeric_integral = zfit.run(argus.numeric_integrate(obs, norm_range=False))
    assert pytest.approx(analytic_integral, 4e-2) == numeric_integral


# register the pdf here and provide sets of working parameter configurations

def gauss_params_factory():
    mu = zfit.Parameter('mu', param1_true)
    sigma = zfit.Parameter('sigma', param2_true)
    return {"mu": mu, "sigma": sigma}


tester.register_pdf(pdf_class=zfit.pdf.Gauss, params_factories=gauss_params_factory)
