import numpy as np
import pytest
import tensorflow as tf
import zfit

# Important, do the imports below
from zfit.core.testing import tester

import zfit_physics as zphys

# specify globals here. Do NOT add any TensorFlow but just pure python

def test_Hypatia2():
    obs = zfit.Space("obs1", (-10, 10))
    hypatia2 = zphys.pdf.Hypatia2(mu=0., sigma=2., nl=2., al=2., nr=2., ar=2., lam=-0.1, beta=0.1, zeta=1, obs=obs)
    assert not any(np.isnan(hypatia2.pdf(tf.linspace(-10, 10, 100))))

    lower = -10
    upper = 10
    hypatia2_pdf = hypatia2.pdf(tf.linspace(lower, upper, 1000001))
    assert pytest.approx(zfit.run(tf.reduce_mean(hypatia2_pdf) * (upper - lower)), 4e-2) == 1.0

    # Compare with RooFit value
    assert pytest.approx(zfit.run(hypatia2.pdf(2)), 1e-3) == 0.112485


def hypatia2_params_factory():
    mu = zfit.Parameter("mu", 0.)
    sigma = zfit.Parameter("sigma", 2.)
    nl = zfit.Parameter("nl", 2.)
    al = zfit.Parameter("al", 2.)
    nr = zfit.Parameter("nr", 2.)
    ar = zfit.Parameter("ar", 2.)
    lam = zfit.Parameter("lam", -0.1)
    beta = zfit.Parameter("beta", 0.1)
    zeta = zfit.Parameter("zeta", 1.)
    return {"mu": mu, "sigma": sigma, "nl": nl, "al": al, "nr": nr, "ar": ar, "lam": lam, "beta": beta, "zeta": zeta}

tester.register_pdf(pdf_class=zphys.pdf.Hypatia2, params_factories=hypatia2_params_factory)
