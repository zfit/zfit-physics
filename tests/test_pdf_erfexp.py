"""Tests for ErfExp PDF."""

import numpy as np
import pytest
import tensorflow as tf
import zfit
from scipy import integrate, special
from zfit.core.testing import tester

import zfit_physics as zphys

alpha_true = 90.0
beta_true = 0.08
gamma_true = -1.0
n_true = 0.2

alpha_range = (65.0, 90.0)
gamma_range = (-10, 10)
beta_range = (0.01, 10)
n_range = (0.1, 0.5)


def _erfexp_numpy(x, alpha, beta, gamma, n):
    return special.erfc((x - alpha) * beta) * np.exp(-gamma * (np.power(x, n) - np.power(alpha, n)))


erfexp_numpy = np.vectorize(_erfexp_numpy, excluded=["alpha", "beta", "gamma", "n"])


def create_erfexp(alpha, beta, gamma, n, limits):
    obs = zfit.Space("obs1", limits=limits)
    erfexp = zphys.pdf.ErfExp(alpha=alpha, beta=beta, gamma=gamma, n=n, obs=obs, norm=False)
    return erfexp, obs


def test_erfexp_pdf():
    # Test PDF here
    erfexp, _ = create_erfexp(alpha=alpha_true, beta=beta_true, gamma=gamma_true, n=n_true, limits=(50, 130))
    assert erfexp.pdf(90.0).numpy().item() == pytest.approx(
        erfexp_numpy(90.0, alpha=alpha_true, beta=beta_true, gamma=gamma_true, n=n_true), rel=1e-8
    )
    np.testing.assert_allclose(
        erfexp.pdf(tf.range(50.0, 130, 10_000)),
        erfexp_numpy(tf.range(50.0, 130, 10_000), alpha=alpha_true, beta=beta_true, gamma=gamma_true, n=n_true),
        rtol=1e-8,
    )


def test_erfexp_pdf_random_params():
    # Test PDF here in a loop with random parameters
    for _ in range(1000):
        alpha_true = np.random.uniform(*alpha_range)
        beta_true = np.random.uniform(*beta_range)
        gamma_true = np.random.uniform(*gamma_range)
        n_true = np.random.uniform(*n_range)

        erfexp, __ = create_erfexp(alpha=alpha_true, beta=beta_true, gamma=gamma_true, n=n_true, limits=(50, 130))
        np.testing.assert_allclose(
            erfexp.pdf(tf.range(50.0, 130, 10_000)),
            erfexp_numpy(tf.range(50.0, 130, 10_000), alpha=alpha_true, beta=beta_true, gamma=gamma_true, n=n_true),
            rtol=1e-5,
        )


def test_erfexp_integral():
    # Test CDF and integral here
    erfexp, obs = create_erfexp(alpha=alpha_true, beta=beta_true, gamma=gamma_true, n=n_true, limits=(50, 130))
    full_interval_numeric = erfexp.numeric_integrate(obs, norm=False).numpy()
    true_integral = 71.18838
    numpy_full_integral = integrate.quad(erfexp_numpy, 50, 130, args=(alpha_true, beta_true, gamma_true, n_true))[0]
    assert full_interval_numeric == pytest.approx(true_integral, 1e-7)
    assert full_interval_numeric == pytest.approx(numpy_full_integral, 1e-7)

    numeric_integral = erfexp.numeric_integrate(limits=(80, 100), norm=False).numpy()
    numpy_integral = integrate.quad(erfexp_numpy, 80, 100, args=(alpha_true, beta_true, gamma_true, n_true))[0]
    assert numeric_integral == pytest.approx(numpy_integral, 1e-7)


# register the pdf here and provide sets of working parameter configurations
def erfexp_params_factory():
    alpha = zfit.Parameter("alpha", alpha_true)
    beta = zfit.Parameter("beta", beta_true)
    gamma = zfit.Parameter("gamma", gamma_true)
    n = zfit.Parameter("n", n_true)

    return {"alpha": alpha, "beta": beta, "gamma": gamma, "n": n}


tester.register_pdf(pdf_class=zphys.pdf.ErfExp, params_factories=erfexp_params_factory)
