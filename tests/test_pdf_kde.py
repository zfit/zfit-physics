"""Example test for a pdf or function."""
import numpy as np
import pytest
import tensorflow as tf
import zfit
from zfit.core.testing import tester

import zfit_physics as zphys

# specify globals here. Do NOT add any TensorFlow but just pure python
param1_true = 0.3
param2_true = 1.2


@pytest.mark.skip  # TODO: remove from package
def test_simple_kde_3d():
    # test special properties  here
    data = np.random.normal(size=(100, 3))
    sigma = zfit.Parameter("sigma", 0.5)

    sigmas = [sigma, 1.0, 2]
    lower = ((-5, -5, -5),)
    upper = ((5, 5, 5),)
    obs = zfit.Space(["obs1", "obs2", "obs3"], limits=(lower, upper))

    kde = zphys.unstable.pdf.GaussianKDE(data=data, bandwidth=sigmas, obs=obs)

    probs = kde.pdf(x=data + 0.03)
    probs_np = probs.numpy()
    from zfit.loss import UnbinnedNLL

    data = np.random.normal(size=(100, 3))

    data = zfit.Data.from_tensor(tensor=data, obs=obs)
    nll = UnbinnedNLL(model=kde, data=data)

    minimizer = zfit.minimize.Minuit()

    minimum = minimizer.minimize(loss=nll)
    assert minimum.converged


@pytest.mark.skip  # TODO: remove from package
def test_simple_kde_1d():
    # test special properties  here
    # zfit.settings.options['numerical_grad'] = True
    data = tf.random.normal(shape=(100, 1))
    # data = np.random.normal(size=(100, 1))
    sigma = zfit.Parameter("sigma", 0.5)
    lower = ((-5,),)
    upper = ((5,),)
    obs = zfit.Space(["obs1"], limits=(lower, upper))

    kde = zphys.unstable.pdf.GaussianKDE(data=data, bandwidth=sigma, obs=obs)

    probs = kde.pdf(x=data + 0.03)

    from zfit.loss import UnbinnedNLL

    data = tf.random.normal(shape=(100, 1))
    data = zfit.Data.from_tensor(tensor=data, obs=obs)
    nll = UnbinnedNLL(model=kde, data=data)

    minimizer = zfit.minimize.Minuit()

    minimum = minimizer.minimize(loss=nll)
    assert minimum.converged


# register the pdf here and provide sets of working parameter configurations


def _kde_params_factory():
    data = np.random.normal(size=(100, 3))
    sigmas = [0.5, 1.0, 2]
    lower = ((-5, -5, -5),)
    upper = ((5, 5, 5),)
    obs = zfit.Space(["obs1", "obs2", "obs3"], limits=(lower, upper))
    return {"data": data, "sigma": sigmas, "obs": obs}


tester.register_pdf(pdf_class=zphys.unstable.pdf.GaussianKDE, params_factories=_kde_params_factory())
