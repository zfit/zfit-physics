"""Example test for a pdf or function"""

import zfit
# Important, do the imports below
from zfit.core.testing import setup_function, teardown_function, tester

import zfit_physics as zphys

# specify globals here. Do NOT add any TensorFlow but just pure python
param1_true = 0.3
param2_true = 1.2


def test_special_property1():
    # test special properties  here
    assert True


# register the pdf here and provide sets of working parameter configurations

def _gauss_params_factory():
    mu_ = zfit.Parameter('mu_cb', param1_true)
    sigma_ = zfit.Parameter('sigma_cb', param2_true)
    return {"mu": mu_, "sigma": sigma_}


tester.register_pdf(pdf_class=zfit.pdf.Gauss, params_factories=_gauss_params_factory())
