"""Tests for Cruijff PDF."""

import numpy as np
import pytest
import tensorflow as tf
import zfit
from numba_stats import cruijff as cruijff_numba
from scipy import integrate
from zfit.core.testing import tester

import zfit_physics as zphys

mu_true = 90.0
sigmal_true = 5.0
alphal_true = 3.0
sigmar_true = 10.0
alphar_true = 2.0


def create_cruijff(mu, sigmal, alphal, sigmar, alphar, limits):
    obs = zfit.Space("obs1", limits=limits)
    cruijff = zphys.pdf.Cruijff(mu=mu, sigmal=sigmal, alphal=alphal, sigmar=sigmar, alphar=alphar, obs=obs)
    return cruijff, obs


def test_cruijff_pdf():
    # Test PDF here
    cruijff, _ = create_cruijff(
        mu=mu_true, sigmal=sigmal_true, alphal=alphal_true, sigmar=sigmar_true, alphar=alphar_true, limits=(50, 130)
    )
    assert cruijff.pdf(90.0, norm=False).numpy() == pytest.approx(
        cruijff_numba.density(
            90.0,
            beta_left=alphal_true,
            beta_right=alphar_true,
            loc=mu_true,
            scale_left=sigmal_true,
            scale_right=sigmar_true,
        ).item(),
        rel=1e-8,
    )
    assert cruijff.pdf(90.0).numpy() == pytest.approx(
        cruijff_numba.density(
            90.0,
            beta_left=alphal_true,
            beta_right=alphar_true,
            loc=mu_true,
            scale_left=sigmal_true,
            scale_right=sigmar_true,
        ).item()
        / 67.71494,
        rel=1e-7,
    )
    np.testing.assert_allclose(
        cruijff.pdf(tf.range(50.0, 130, 10_000), norm=False),
        cruijff_numba.density(
            tf.range(50.0, 130, 10_000).numpy(),
            beta_left=alphal_true,
            beta_right=alphar_true,
            loc=mu_true,
            scale_left=sigmal_true,
            scale_right=sigmar_true,
        ),
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        cruijff.pdf(tf.range(50.0, 130, 10_000)),
        cruijff_numba.density(
            tf.range(50.0, 130, 10_000).numpy(),
            beta_left=alphal_true,
            beta_right=alphar_true,
            loc=mu_true,
            scale_left=sigmal_true,
            scale_right=sigmar_true,
        )
        / 67.71494,
        rtol=1e-8,
    )
    assert cruijff.pdf(tf.range(50.0, 130, 10_000)) <= cruijff.pdf(90.0)


def test_cruijff_integral():
    # Test CDF and integral here
    cruijff, obs = create_cruijff(
        mu=mu_true, sigmal=sigmal_true, alphal=alphal_true, sigmar=sigmar_true, alphar=alphar_true, limits=(50, 130)
    )
    full_interval_numeric = cruijff.numeric_integrate(obs, norm=False).numpy()
    true_integral = 67.71494
    numba_stats_full_integral = integrate.quad(
        cruijff_numba.density, 50, 130, args=(alphal_true, alphar_true, mu_true, sigmal_true, sigmar_true)
    )[0]
    assert full_interval_numeric == pytest.approx(true_integral, 1e-7)
    assert full_interval_numeric == pytest.approx(numba_stats_full_integral, 1e-7)

    numeric_integral = cruijff.numeric_integrate(limits=(80, 100), norm=False).numpy()
    numba_stats_integral = integrate.quad(
        cruijff_numba.density, 80, 100, args=(alphal_true, alphar_true, mu_true, sigmal_true, sigmar_true)
    )[0]
    assert numeric_integral == pytest.approx(numba_stats_integral, 1e-7)


# register the pdf here and provide sets of working parameter configurations
def cruijff_params_factory():
    mu = zfit.Parameter("mu", mu_true)
    sigmal = zfit.Parameter("sigmal", sigmal_true)
    alphal = zfit.Parameter("alphal", alphal_true)
    sigmar = zfit.Parameter("sigmar", sigmar_true)
    alphar = zfit.Parameter("alphar", alphar_true)

    return {"mu": mu, "sigmal": sigmal, "alphal": alphal, "sigmar": sigmar, "alphar": alphar}


tester.register_pdf(pdf_class=zphys.pdf.Cruijff, params_factories=cruijff_params_factory)
