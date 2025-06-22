"""Tests for CMSShape PDF."""

import numpy as np
import pytest
import tensorflow as tf
import zfit
from numba_stats import cmsshape as cmsshape_numba
# Important, do the imports below
from zfit.core.testing import tester

import zfit_physics as zphys

# specify globals here. Do NOT add any TensorFlow but just pure python
m_true = 90.0
beta_true = 0.2
gamma_true = 0.3


def create_cmsshape(m, beta, gamma, limits):
    obs = zfit.Space("obs1", limits)
    cmsshape = zphys.pdf.CMSShape(m=m, beta=beta, gamma=gamma, obs=obs)
    return cmsshape, obs


def test_cmsshape_pdf():
    # Test PDF here
    cmsshape, _ = create_cmsshape(m=m_true, beta=beta_true, gamma=gamma_true, limits=(50, 130))
    assert zfit.run(cmsshape.pdf(90.0)) == pytest.approx(
        cmsshape_numba.pdf(90.0, beta=beta_true, gamma=gamma_true, loc=m_true).item(), rel=1e-5
    )
    np.testing.assert_allclose(
        cmsshape.pdf(tf.range(50.0, 130, 10_000)),
        cmsshape_numba.pdf(tf.range(50.0, 130, 10_000).numpy(), beta=beta_true, gamma=gamma_true, loc=m_true),
        rtol=1e-5,
    )
    assert cmsshape.pdf(tf.range(50.0, 130, 10_000)) <= cmsshape.pdf(90.0)

    sample = cmsshape.sample(1000)
    tf.debugging.assert_all_finite(sample.value(), "Some samples from the cmsshape PDF are NaN or infinite")
    assert sample.n_events == 1000
    assert all(tf.logical_and(50 <= sample.value(), sample.value() <= 130))


def test_cmsshape_integral():
    # Test CDF and integral here
    cmsshape, obs = create_cmsshape(m=m_true, beta=beta_true, gamma=gamma_true, limits=(50, 130))
    full_interval_analytic = zfit.run(cmsshape.analytic_integrate(obs, norm=False))
    full_interval_numeric = zfit.run(cmsshape.numeric_integrate(obs, norm=False))
    true_integral = 0.99999
    numba_stats_full_integral = cmsshape_numba.cdf(
        130, beta=beta_true, gamma=gamma_true, loc=m_true
    ) - cmsshape_numba.cdf(50, beta=beta_true, gamma=gamma_true, loc=m_true)
    assert full_interval_analytic == pytest.approx(true_integral, 1e-5)
    assert full_interval_numeric == pytest.approx(true_integral, 1e-5)
    assert full_interval_analytic == pytest.approx(numba_stats_full_integral, 1e-8)
    assert full_interval_numeric == pytest.approx(numba_stats_full_integral, 1e-8)

    analytic_integral = zfit.run(cmsshape.analytic_integrate(limits=(80, 100), norm=False))
    numeric_integral = zfit.run(cmsshape.numeric_integrate(limits=(80, 100), norm=False))
    numba_stats_integral = cmsshape_numba.cdf(100, beta=beta_true, gamma=gamma_true, loc=m_true) - cmsshape_numba.cdf(
        80, beta=beta_true, gamma=gamma_true, loc=m_true
    )
    assert analytic_integral == pytest.approx(numeric_integral, 1e-8)
    assert analytic_integral == pytest.approx(numba_stats_integral, 1e-8)


# register the pdf here and provide sets of working parameter configurations
def cmsshape_params_factory():
    m = zfit.Parameter("m", m_true)
    beta = zfit.Parameter("beta", beta_true)
    gamma = zfit.Parameter("gamma", gamma_true)

    return {"m": m, "beta": beta, "gamma": gamma}


tester.register_pdf(pdf_class=zphys.pdf.CMSShape, params_factories=cmsshape_params_factory)
