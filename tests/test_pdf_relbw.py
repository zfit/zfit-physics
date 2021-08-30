"""Tests for relativistic Breit-Wigner PDF."""
import pytest
import tensorflow as tf
import zfit

# Important, do the imports below
from zfit.core.testing import tester

import zfit_physics as zphys

# specify globals here. Do NOT add any TensorFlow but just pure python
m_true = 125.
Gamma_true = 0.05


def create_relbw(m, Gamma, limits):
    obs = zfit.Space("obs1", limits)
    relbw = zphys.pdf.RelativisticBreitWigner(m=m, Gamma=Gamma, obs=obs)
    return relbw, obs


def test_relbw_pdf():
    # Test PDF here
    relbw, _ = create_relbw(m_true, Gamma_true, limits=(0, 200))
    assert zfit.run(relbw.pdf(125.)) == pytest.approx(12.732396211295313, rel=1e-4)
    assert relbw.pdf(tf.range(0., 200, 10_000)) <= relbw.pdf(125.)

    sample = relbw.sample(1000)
    tf.debugging.assert_all_finite(sample, 'Some samples from the relbw PDF are NaN or infinite')
    assert sample.n_events == 1000
    assert all(tf.logical_and(0 <= sample, sample <= 200))


def test_relbw_integral():
    # Test CDF and integral here
    relbw, obs = create_relbw(m_true, Gamma_true, limits=(0, 200))

    analytic_integral = zfit.run(relbw.analytic_integrate(obs, norm_range=False))[0]
    numeric_integral = zfit.run(relbw.numeric_integrate(obs, norm_range=False))[0]
    assert analytic_integral == pytest.approx(1., 1e-4)
    assert numeric_integral == pytest.approx(1., 2e-3)


# register the pdf here and provide sets of working parameter configurations
def relbw_params_factory():
    m = zfit.Parameter("m", m_true)
    Gamma = zfit.Parameter("Gamma", Gamma_true)
    return {"m": m, "Gamma": Gamma}


tester.register_pdf(pdf_class=zphys.pdf.RelativisticBreitWigner, params_factories=relbw_params_factory)