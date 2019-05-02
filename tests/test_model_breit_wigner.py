"""Example test for a pdf or function"""
import pytest
import zfit
import numpy as np
# Important, do the imports below
from zfit.core.testing import setup_function, teardown_function, tester

import zfit_physics as zphys

# specify globals here. Do NOT add any TensorFlow but just pure python
param1_true = 0.3
param2_true = 1.2
obs1 = zfit.Space("obs1", limits=(-4, 5))


def test_bw_pdf():
    # test special properties  here
    bw = zphys.unstable.pdf.RelativisticBreitWignerSquared(obs=obs1, mres=1., wres=0.3)

    integral = bw.integrate(limits=obs1)
    assert pytest.approx(1., rel=1e-3) == zfit.run(integral)


def test_bw_func():
    # test special properties  here
    bw = zphys.unstable.func.RelativisticBreitWigner(obs=obs1, mres=1., wres=0.3)

    vals = bw.func(x=np.random.uniform(size=(100, 1)))
    assert 100 == len(zfit.run(vals))


# register the pdf here and provide sets of working parameter configurations

def _bw_params_factory():
    mres = zfit.Parameter('mres', param1_true)
    wres = zfit.Parameter('wres', param2_true)
    return {"mres": mres, "wres": wres}

# tester.register_func(func_class=zfit.pdf.Gauss, params_factories=_bw_params_factory())
