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

def _bw_params_factory():
    mres = zfit.Parameter('mres', param1_true)
    wres = zfit.Parameter('wres', param2_true)
    return {"mres": mres, "mres": mres}


# tester.register_func(func_class=zfit.pdf.Gauss, params_factories=_bw_params_factory())
