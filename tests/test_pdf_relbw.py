"""Tests for relativistic Breit-Wigner PDF."""
import pytest
import tensorflow as tf
import zfit

# Important, do the imports below
from zfit.core.testing import tester

import zfit_physics as zphys

# specify globals here. Do NOT add any TensorFlow but just pure python
m_true = 125.0
gamma_true = 1.5


def create_relbw(m, gamma, limits):
    obs = zfit.Space("obs1", limits)
    relbw = zphys.pdf.RelativisticBreitWigner(m=m, gamma=gamma, obs=obs)
    return relbw, obs


def test_relbw_pdf():
    # Test PDF here
    relbw, _ = create_relbw(m_true, gamma_true, limits=(0, 200))
    assert zfit.run(relbw.pdf(125.0)) == pytest.approx(0.4249, rel=1e-4)
    assert relbw.pdf(tf.range(0.0, 200, 10_000)) <= relbw.pdf(125.0)

    sample = relbw.sample(1000)
    tf.debugging.assert_all_finite(sample.value(), "Some samples from the relbw PDF are NaN or infinite")
    assert sample.n_events == 1000
    assert all(tf.logical_and(0 <= sample.value(), sample.value() <= 200))


def test_relbw_integral():
    # Test CDF and integral here
    relbw, obs = create_relbw(m_true, gamma_true, limits=(0, 200))
    full_interval_analytic = zfit.run(relbw.analytic_integrate(obs, norm_range=False))
    full_interval_numeric = zfit.run(relbw.numeric_integrate(obs, norm_range=False))
    true_integral = 0.99888
    assert full_interval_analytic == pytest.approx(true_integral, 1e-4)
    assert full_interval_numeric == pytest.approx(true_integral, 1e-2)

    analytic_integral = zfit.run(relbw.analytic_integrate(limits=(50, 100), norm_range=False))
    numeric_integral = zfit.run(relbw.numeric_integrate(limits=(50, 100), norm_range=False))
    assert analytic_integral == pytest.approx(numeric_integral, 0.01)


# register the pdf here and provide sets of working parameter configurations
def relbw_params_factory():
    m = zfit.Parameter("m", m_true)
    gamma = zfit.Parameter("gamma", gamma_true)
    return {"m": m, "gamma": gamma}


tester.register_pdf(pdf_class=zphys.pdf.RelativisticBreitWigner, params_factories=relbw_params_factory)
