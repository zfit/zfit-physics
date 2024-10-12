"""Tests for CMSShape PDF."""

import numpy as np
import pytest
import tensorflow as tf
import zfit
from numba_stats import tsallis as tsallis_numba
# Important, do the imports below
from zfit.core.testing import tester

import zfit_physics as zphys

# specify globals here. Do NOT add any TensorFlow but just pure python
m_true = 90.0
t_true = 10.0
n_true = 3.0


def create_tsallis(m, t, n, limits):
    obs = zfit.Space("obs1", limits)
    tsallis = zphys.pdf.Tsallis(m=m, t=t, n=n, obs=obs)
    return tsallis, obs


def test_tsallis_pdf():
    # Test PDF here
    tsallis, _ = create_tsallis(m=m_true, t=t_true, n=n_true, limits=(0, 150))
    assert tsallis.pdf(90.0, norm=False).numpy().item() == pytest.approx(
        tsallis_numba.pdf(90.0, m=m_true, t=t_true, n=n_true), 1e-5
    )
    np.testing.assert_allclose(
        tsallis.pdf(tf.range(0.0, 150, 10_000), norm=False),
        tsallis_numba.pdf(tf.range(0.0, 150, 10_000).numpy(), m=m_true, t=t_true, n=n_true),
        rtol=1e-5,
    )

    sample = tsallis.sample(1000)
    assert all(np.isfinite(sample.value())), "Some samples from the tsallis PDF are NaN or infinite"
    assert sample.n_events == 1000
    assert all(tf.logical_and(0 <= sample.value(), sample.value() <= 150))


def test_tsallis_integral():
    # Test CDF and integral here
    tsallis, obs = create_tsallis(m=m_true, t=t_true, n=n_true, limits=(0, 150))
    full_interval_analytic = zfit.run(tsallis.analytic_integrate(obs, norm=False))
    full_interval_numeric = zfit.run(tsallis.numeric_integrate(obs, norm=False))
    true_integral = 0.835415
    numba_stats_full_integral = tsallis_numba.cdf(150, m=m_true, t=t_true, n=n_true) - tsallis_numba.cdf(
        0, m=m_true, t=t_true, n=n_true
    )
    assert full_interval_analytic == pytest.approx(true_integral, 1e-5)
    assert full_interval_numeric == pytest.approx(true_integral, 1e-5)
    assert full_interval_analytic == pytest.approx(numba_stats_full_integral, 1e-8)
    assert full_interval_numeric == pytest.approx(numba_stats_full_integral, 1e-8)

    analytic_integral = zfit.run(tsallis.analytic_integrate(limits=(20, 60), norm=False))
    numeric_integral = zfit.run(tsallis.numeric_integrate(limits=(20, 60), norm=False))
    numba_stats_integral = tsallis_numba.cdf(60, m=m_true, t=t_true, n=n_true) - tsallis_numba.cdf(
        20, m=m_true, t=t_true, n=n_true
    )
    assert analytic_integral == pytest.approx(numeric_integral, 1e-8)
    assert analytic_integral == pytest.approx(numba_stats_integral, 1e-8)


# register the pdf here and provide sets of working parameter configurations
def tsallis_params_factory():
    m = zfit.Parameter("m", m_true)
    t = zfit.Parameter("t", t_true)
    n = zfit.Parameter("n", n_true)

    return {"m": m, "t": t, "n": n}


tester.register_pdf(pdf_class=zphys.pdf.Tsallis, params_factories=tsallis_params_factory)
