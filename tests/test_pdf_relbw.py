"""Tests for relativistic Breit-Wigner PDF."""
import numpy as np
import pytest
import tensorflow as tf
import zfit

# Important, do the imports below
from zfit.core.testing import tester

import zfit_physics as zphys

# specify globals here. Do NOT add any TensorFlow but just pure python
param1_true = 0.3
param2_true = 1.2


def test_pdf():
    # Test PDF values here
    obs = zfit.Space("obs1", (0, 200))

    relbw = zphys.pdf.RelativisticBreitWigner(m=125.0, Gamma=0.05, obs=obs)
    assert zfit.run(relbw.pdf(125.0)) == pytest.approx(12.7, rel=1e-3)

    # analytic_integral = zfit.run(relbw.analytic_integrate(obs, norm_range=False))
    # numeric_integral = zfit.run(relbw.numeric_integrate(obs, norm_range=False))
    # assert pytest.approx(analytic_integral, 4e-2) == numeric_integral


# register the pdf here and provide sets of working parameter configurations


def relbw_params_factory():
    m = zfit.Parameter("m", 4.5)
    Gamma = zfit.Parameter("Gamma", -2.3)
    return {"m": m, "Gamma": Gamma}


tester.register_pdf(pdf_class=zphys.pdf.RelativisticBreitWigner, params_factories=relbw_params_factory)
