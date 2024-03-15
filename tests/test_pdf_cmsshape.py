"""Tests for CMSShape PDF."""
import pytest
import tensorflow as tf
import zfit
from numba_stats import cmsshape as cmsshape_numba

# Important, do the imports below
from zfit.core.testing import tester

import zfit_physics as zphys

# specify globals here. Do NOT add any TensorFlow but just pure python
beta_true = 0.2
gamma_true = 0.3
m_true = 90.0


def create_cmsshape(gamma, beta, m, limits):
    obs = zfit.Space("obs1", limits)
    cmsshape = zphys.pdf.CMSShape(gamma=gamma, beta=beta, m=m, obs=obs)
    return cmsshape, obs


def test_cmsshape_pdf():
    # Test PDF here
    cmsshape, _ = create_cmsshape(gamma=gamma_true, beta=beta_true, m=m_true, limits=(50, 130))
    assert zfit.run(cmsshape.pdf(90.0)) == pytest.approx(
        cmsshape_numba.pdf(90.0, beta=beta_true, gamma=gamma_true, loc=m_true).item(), rel=1e-4
    )
    assert cmsshape.pdf(tf.range(50.0, 130, 10_000)) <= cmsshape.pdf(90.0)

    sample = cmsshape.sample(1000)
    tf.debugging.assert_all_finite(sample.value(), "Some samples from the cmsshape PDF are NaN or infinite")
    assert sample.n_events == 1000
    assert all(tf.logical_and(50 <= sample.value(), sample.value() <= 130))


def test_cmsshape_integral():
    # Test CDF and integral here
    cmsshape, obs = create_cmsshape(gamma=gamma_true, beta=beta_true, m=m_true, limits=(50, 130))
    full_interval_analytic = zfit.run(cmsshape.analytic_integrate(obs, norm_range=False))
    full_interval_numeric = zfit.run(cmsshape.numeric_integrate(obs, norm_range=False))
    true_integral = 0.99999
    assert full_interval_analytic == pytest.approx(true_integral, 1e-4)
    assert full_interval_numeric == pytest.approx(true_integral, 1e-2)

    analytic_integral = zfit.run(cmsshape.analytic_integrate(limits=(80, 100), norm_range=False))
    numeric_integral = zfit.run(cmsshape.numeric_integrate(limits=(80, 100), norm_range=False))
    assert analytic_integral == pytest.approx(numeric_integral, 0.01)


# register the pdf here and provide sets of working parameter configurations
def cmsshape_params_factory():
    beta = zfit.Parameter("beta", beta_true)
    gamma = zfit.Parameter("gamma", gamma_true)
    m = zfit.Parameter("m", m_true)
    return {"beta": beta, "gamma": gamma, "m": m}


tester.register_pdf(pdf_class=zphys.pdf.CMSShape, params_factories=cmsshape_params_factory)
