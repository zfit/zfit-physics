"""Tests for Novosibirsk PDF."""

import numpy as np
import pytest
import ROOT
import tensorflow as tf
import zfit
from zfit.core.testing import tester

import zfit_physics as zphys

mu_true = 90.0
sigma_true = 10.0
lambd_true = 0.5


def create_novosibirsk(mu, sigma, lambd, limits):
    obs = zfit.Space("obs1", limits)
    novosibirsk = zphys.pdf.Novosibirsk(mu=mu, sigma=sigma, lambd=lambd, obs=obs)
    return novosibirsk, obs


def create_and_eval_root_novosibirsk_and_integral(mu, sigma, lambd, limits, x, lower, upper):
    obs = ROOT.RooRealVar("obs", "obs", *limits)
    peak = ROOT.RooRealVar("peak", "peak", mu)
    width = ROOT.RooRealVar("width", "width", sigma)
    tail = ROOT.RooRealVar("tail", "tail", lambd)
    novosibirsk = ROOT.RooNovosibirsk("novosibirsk", "novosibirsk", obs, peak, width, tail)

    out = np.zeros_like(x)
    for i, xi in enumerate(x):
        obs.setVal(xi)
        out[i] = novosibirsk.getVal(ROOT.RooArgSet(obs))

    obs.setRange("integrationRange", lower, upper)
    integral = novosibirsk.createIntegral(ROOT.RooArgSet(obs), ROOT.RooFit.Range("integrationRange")).getVal()
    return out, integral


def test_novosibirsk_pdf():
    # Teat PDF here
    novosibirsk, _ = create_novosibirsk(mu=mu_true, sigma=sigma_true, lambd=lambd_true, limits=(50, 130))
    novosibirsk_root_90 = create_and_eval_root_novosibirsk_and_integral(
        mu=mu_true,
        sigma=sigma_true,
        lambd=lambd_true,
        limits=(50, 130),
        x=np.array([90.0]),
        lower=50,
        upper=130,
    )[0]
    assert novosibirsk.pdf(90.0).numpy() == pytest.approx(novosibirsk_root_90, rel=1e-5)
    test_values = tf.range(50.0, 130, 10_000)
    novosibirsk_root_test_values = create_and_eval_root_novosibirsk_and_integral(
        mu=mu_true,
        sigma=sigma_true,
        lambd=lambd_true,
        limits=(50, 130),
        x=test_values,
        lower=50,
        upper=130,
    )[0]
    np.testing.assert_allclose(novosibirsk.pdf(test_values).numpy(), novosibirsk_root_test_values, rtol=1e-5)
    assert novosibirsk.pdf(test_values) <= novosibirsk.pdf(90.0)
    sample = novosibirsk.sample(1000)
    assert all(np.isfinite(sample.value())), "Some samples from the Novosibirsk PDF are NaN or infinite"
    assert sample.n_events == 1000
    assert all(tf.logical_and(50 <= sample.value(), sample.value() <= 130))


def test_novosibirsk_integral():
    # Test CDF and integral here
    novosibirsk, obs = create_novosibirsk(mu=mu_true, sigma=sigma_true, lambd=lambd_true, limits=(50, 130))
    full_interval_analytic = novosibirsk.analytic_integrate(obs, norm=False).numpy()
    full_interval_numeric = novosibirsk.numeric_integrate(obs, norm=False).numpy()
    true_integral = 23.021161
    root_integral = create_and_eval_root_novosibirsk_and_integral(
        mu=mu_true,
        sigma=sigma_true,
        lambd=lambd_true,
        limits=(50, 130),
        x=np.array([50, 130]),
        lower=50,
        upper=130,
    )[1]
    assert full_interval_analytic == pytest.approx(true_integral, 1e-5)
    assert full_interval_numeric == pytest.approx(true_integral, 1e-5)
    assert full_interval_analytic == pytest.approx(root_integral, 1e-5)
    assert full_interval_numeric == pytest.approx(root_integral, 1e-5)

    analytic_integral = novosibirsk.analytic_integrate(limits=(80, 100), norm=False).numpy()
    numeric_integral = novosibirsk.numeric_integrate(limits=(80, 100), norm=False).numpy()
    root_integral = create_and_eval_root_novosibirsk_and_integral(
        mu=mu_true,
        sigma=sigma_true,
        lambd=lambd_true,
        limits=(50, 130),
        x=np.array([80, 100]),
        lower=80,
        upper=100,
    )[1]
    assert analytic_integral == pytest.approx(numeric_integral, 1e-5)
    assert analytic_integral == pytest.approx(root_integral, 1e-5)


# register the pdf here and provide sets of working parameter configurations
def novosibirsk_params_factory():
    mu = zfit.Parameter("mu", mu_true)
    sigma = zfit.Parameter("sigma", sigma_true)
    lambd = zfit.Parameter("lambd", lambd_true)
    return {"mu": mu, "sigma": sigma, "lambd": lambd}


tester.register_pdf(pdf_class=zphys.pdf.Novosibirsk, params_factories=novosibirsk_params_factory)
