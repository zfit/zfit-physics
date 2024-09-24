"""Tests for Hypatia2 PDF."""
import numpy as np
import pytest

ROOT = pytest.importorskip("ROOT")
import tensorflow as tf
import zfit
from zfit.core.testing import tester

import zfit_physics as zphys

mu_true = 0.0
sigma_true = 1.0
lambd_true = -1.0
zeta_true = 1.0
beta_true = -0.01
al_true = 50.0
nl_true = 1.5
ar_true = 1.0
nr_true = 0.1


def create_hypatia2(mu, sigma, lambd, zeta, beta, al, nl, ar, nr, limits):
    obs = zfit.Space("obs1", limits)
    hypatia2 = zphys.pdf.Hypatia2(mu=mu, sigma=sigma, lambd=lambd, zeta=zeta, beta=beta, al=al, nl=nl, ar=ar, nr=nr, obs=obs)
    return hypatia2, obs


def create_and_eval_root_hypatia2_and_integral(mu, sigma, lambd, zeta, beta, al, nl, ar, nr, limits, x, lower, upper):
    obs = ROOT.RooRealVar("obs1", "obs1", *limits)
    lambd = ROOT.RooRealVar("lambda", "lambda", lambd)
    zeta = ROOT.RooRealVar("zeta", "zeta", zeta)
    beta = ROOT.RooRealVar("beta", "beta", beta)
    argSigma = ROOT.RooRealVar("argSigma", "argSigma", sigma)
    mu = ROOT.RooRealVar("mu", "mu", mu)
    a = ROOT.RooRealVar("a", "a", al)
    n = ROOT.RooRealVar("n", "n", nl)
    a2 = ROOT.RooRealVar("a2", "a2", ar)
    n2 = ROOT.RooRealVar("n2", "n2", nr)
    hypatia2 = ROOT.RooHypatia2("hypatia2", "hypatia2", obs, lambd, zeta, beta, argSigma, mu, a, n, a2, n2)

    out = np.zeros_like(x)
    for i, xi in enumerate(x):
        obs.setVal(xi)
        out[i] = hypatia2.getVal(ROOT.RooArgSet(obs))

    obs.setRange("integrationRange", lower, upper)
    integral = hypatia2.createIntegral(ROOT.RooArgSet(obs), ROOT.RooFit.Range("integrationRange")).getVal()
    return out, integral


def test_hypatia2_pdf():
    # Test PDF here
    hypatia2, _ = create_hypatia2(
        mu=mu_true, sigma=sigma_true, lambd=lambd_true, zeta=zeta_true, beta=beta_true, al=al_true, nl=nl_true, ar=ar_true, nr=nr_true, limits=(-10, 10)
    )
    hypatia2_root = create_and_eval_root_hypatia2_and_integral(
        mu=mu_true, sigma=sigma_true, lambd=lambd_true, zeta=zeta_true, beta=beta_true, al=al_true, nl=nl_true, ar=ar_true, nr=nr_true, limits=(-10, 10), x=np.array([0.0]), lower=-10, upper=10
    )[0]
    assert hypatia2.pdf(0.0).numpy() == pytest.approx(hypatia2_root, rel=1e-4)
    test_values = tf.range(-10.0, 10, 10_000)
    hypatia2_root_test_values = create_and_eval_root_hypatia2_and_integral(
        mu=mu_true, sigma=sigma_true, lambd=lambd_true, zeta=zeta_true, beta=beta_true, al=al_true, nl=nl_true, ar=ar_true, nr=nr_true, limits=(-10, 10), x=test_values, lower=-10, upper=10
    )[0]
    np.testing.assert_allclose(hypatia2.pdf(test_values).numpy(), hypatia2_root_test_values, rtol=1e-4)
    assert hypatia2.pdf(test_values) <= hypatia2.pdf(0.0)
    sample = hypatia2.sample(1000)
    assert all(np.isfinite(sample.numpy())), "Some samples from the Hypatia2 PDF are NaN or infinite"
    assert sample.n_events == 1000
    assert all(tf.logical_and(-10 <= sample.value(), sample.value() <= 10))


def test_hypatia2_integral():
    # Test CDF and integral here
    hypatia2, obs = create_hypatia2(
        mu=mu_true, sigma=sigma_true, lambd=lambd_true, zeta=zeta_true, beta=beta_true, al=al_true, nl=nl_true, ar=ar_true, nr=nr_true, limits=(-10, 10)
    )
    full_interval_numeric = hypatia2.numeric_integrate(obs, norm=False).numpy()
    root_integral = create_and_eval_root_hypatia2_and_integral(
        mu=mu_true, sigma=sigma_true, lambd=lambd_true, zeta=zeta_true, beta=beta_true, al=al_true, nl=nl_true, ar=ar_true, nr=nr_true, limits=(-10, 10), x=np.array([-10, 10]), lower=-10, upper=10
    )[1]
    assert full_interval_numeric == pytest.approx(root_integral, 1e-4)

    numeric_integral = hypatia2.numeric_integrate(limits=(-2, 4), norm=False).numpy()
    root_integral = create_and_eval_root_hypatia2_and_integral(
        mu=mu_true, sigma=sigma_true, lambd=lambd_true, zeta=zeta_true, beta=beta_true, al=al_true, nl=nl_true, ar=ar_true, nr=nr_true, limits=(-10, 10), x=np.array([-2, 4]), lower=-2, upper=4
    )[1]
    assert numeric_integral == pytest.approx(root_integral, 1e-4)


def hypatia2_params_factory():
    mu = zfit.Parameter("mu", mu_true)
    sigma = zfit.Parameter("sigma", sigma_true)
    lambd = zfit.Parameter("lambd", lambd_true)
    zeta = zfit.Parameter("zeta", zeta_true)
    beta = zfit.Parameter("beta", beta_true)
    al = zfit.Parameter("al", al_true)
    nl = zfit.Parameter("nl", nl_true)
    ar = zfit.Parameter("ar", ar_true)
    nr = zfit.Parameter("nr", nr_true)
    return {
        "mu": mu,
        "sigma": sigma,
        "lambd": lambd,
        "zeta": zeta,
        "beta": beta,
        "al": al,
        "nl": nl,
        "ar": ar,
        "nr": nr
    }

tester.register_pdf(pdf_class=zphys.pdf.Hypatia2, params_factories=hypatia2_params_factory)
