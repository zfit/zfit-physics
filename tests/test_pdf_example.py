"""Example test for a pdf or function."""

import zfit
from zfit.core.testing import tester

import zfit_physics as zphys

# specify globals here. Do NOT add any TensorFlow but just pure python
param1_true = 0.3
param2_true = 1.2


def test_special_property1():
    # test special properties  here
    assert True


# register the pdf here and provide sets of working parameter configurations


def gauss_params_factory():
    mu = zfit.Parameter("mu", param1_true)
    sigma = zfit.Parameter("sigma", param2_true)
    return {"mu": mu, "sigma": sigma}


tester.register_pdf(pdf_class=zfit.pdf.Gauss, params_factories=gauss_params_factory)
