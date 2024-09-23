import numpy as np
import pytest
import tensorflow as tf
import zfit

# Important, do the imports below
from zfit.core.testing import tester

import zfit_physics as zphys

ROOT = pytest.importorskip("ROOT")

# specify globals here. Do NOT add any TensorFlow but just pure python

mu_true=0.0
sigma_true = 2.0
nl_true = 2.0
al_true = 2.0
nr_true = 2.0
ar_true = 2.0
lam_true = -0.1
beta_true = 0.1
zeta_true = 1.0


def create_and_eval_roofit_hypatia2(limits, x):
    obs = ROOT.RooRealVar("obs", "obs", *limits)
    mu = ROOT.RooRealVar("mu", "mu", mu_true)
    sigma = ROOT.RooRealVar("sigma", "sigma", sigma_true)
    nl = ROOT.RooRealVar("nl", "nl", nl_true)
    al = ROOT.RooRealVar("al", "al", al_true)
    nr = ROOT.RooRealVar("nr", "nr", nr_true)
    ar = ROOT.RooRealVar("ar", "ar", ar_true)
    lam = ROOT.RooRealVar("lam", "lam", lam_true)
    beta = ROOT.RooRealVar("beta", "beta", beta_true)
    zeta = ROOT.RooRealVar("zeta", "zeta", zeta_true)

    hypatia2 = ROOT.RooHypatia2("hypatia2", "hypatia2", obs, lam, zeta, beta, sigma, mu, al, nl, ar, nr)

    out = np.zeros_like(x)
    for i, xi in enumerate(x):
        obs.setVal(xi)
        out[i] = hypatia2.getVal(ROOT.RooArgSet(obs))

    return out


def test_Hypatia2():
    limits = (-10, 10)
    obs = zfit.Space("obs1", limits)
    hypatia2 = zphys.pdf.Hypatia2(mu=mu_true, sigma=sigma_true, nl=nl_true, al=al_true, nr=nr_true, ar=ar_true, lam=lam_true, beta=beta_true, zeta=zeta_true, obs=obs)
    assert not any(np.isnan(hypatia2.pdf(tf.linspace(-10, 10, 100))))

    lower = -10
    upper = 10
    x = tf.linspace(lower, upper, 100001)
    hypatia2_pdf = hypatia2.pdf(x)
    assert pytest.approx(zfit.run(tf.reduce_mean(hypatia2_pdf) * (upper - lower)), 1e-3) == 1.0

    # Compare with RooFit value
    roofit_hypatia2_pdf = create_and_eval_roofit_hypatia2(limits, x)
    np.testing.assert_allclose(hypatia2_pdf.numpy(), roofit_hypatia2_pdf, rtol=1e-5)


def hypatia2_params_factory():
    mu = zfit.Parameter("mu", mu_true)
    sigma = zfit.Parameter("sigma", sigma_true)
    nl = zfit.Parameter("nl", nl_true)
    al = zfit.Parameter("al", al_true)
    nr = zfit.Parameter("nr", nr_true)
    ar = zfit.Parameter("ar", ar_true)
    lam = zfit.Parameter("lam", lam_true)
    beta = zfit.Parameter("beta", beta_true)
    zeta = zfit.Parameter("zeta", zeta_true)
    return {"mu": mu, "sigma": sigma, "nl": nl, "al": al, "nr": nr, "ar": ar, "lam": lam, "beta": beta, "zeta": zeta}

tester.register_pdf(pdf_class=zphys.pdf.Hypatia2, params_factories=hypatia2_params_factory)
