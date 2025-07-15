"""Example test for a pdf or function."""

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


def test_standard_argus():
    # test special properties  here
    obs = zfit.Space("obs1", (-2, 6))

    argus = zphys.pdf.Argus(m0=5.0, c=-3.0, p=0.5, obs=obs)
    assert not any(np.isnan(argus.pdf(tf.linspace(0.1, 15.0, 100))))
    lower = 0.0
    upper = 5.0
    argus_pdf = argus.pdf(tf.linspace(lower, upper, 1000001))
    assert pytest.approx(zfit.run(tf.reduce_mean(argus_pdf) * (upper - lower)), 4e-2) == 1.0
    analytic_integral = zfit.run(argus.analytic_integrate(obs, norm=False))
    numeric_integral = zfit.run(argus.numeric_integrate(obs, norm=False))
    assert pytest.approx(analytic_integral, 4e-2) == numeric_integral


# register the pdf here and provide sets of working parameter configurations


def argus_params_factory():
    m0 = zfit.Parameter("m0", 4.5)
    c = zfit.Parameter("c", -2.3)
    p = zfit.param.ConstantParameter("p", 0.5)
    return {"m0": m0, "c": c, "p": p}


tester.register_pdf(pdf_class=zphys.pdf.Argus, params_factories=argus_params_factory)


def test_argus_serialization():
    """Test Argus PDF serialization."""
    obs = zfit.Space("x", (-2, 6))
    m0_val = 5.0
    c_val = -3.0
    p_val = 0.5

    argus = zphys.pdf.Argus(m0=m0_val, c=c_val, p=p_val, obs=obs)

    # Test serialization
    pdf_dict = argus.to_dict()
    assert pdf_dict['type'] == 'Argus'

    # Test deserialization
    reconstructed_argus = zphys.pdf.Argus.from_dict(pdf_dict)

    # Verify functionality - test that the PDFs produce the same values
    test_data = tf.linspace(0.1, 4.9, 100)
    original_values = argus.pdf(test_data)
    reconstructed_values = reconstructed_argus.pdf(test_data)

    # Check that values are close (allowing for small numerical differences)
    np.testing.assert_allclose(zfit.run(original_values), zfit.run(reconstructed_values), rtol=1e-10)
