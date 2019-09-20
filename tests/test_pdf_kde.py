"""Example test for a pdf or function"""

import zfit
import numpy as np
# Important, do the imports below
from zfit.core.testing import setup_function, teardown_function, tester

import zfit_physics as zphys

# specify globals here. Do NOT add any TensorFlow but just pure python
param1_true = 0.3
param2_true = 1.2


def test_simple_kde_3d():
    # test special properties  here
    data = np.random.normal(size=(100, 3))
    sigmas = [0.5, 1., 2]
    lower = ((-5, -5, -5),)
    upper = ((5, 5, 5),)
    obs = zfit.Space(["obs1", "obs2", "obs3"], limits=(lower, upper))

    kde = zphys.unstable.pdf.GaussianKDE(data=data, bandwidth=sigmas, obs=obs)

    probs = kde.pdf(x=data + 0.03)
    probs_np = zfit.run(probs)


def test_simple_kde_1d():
    # test special properties  here
    data = np.random.normal(size=(100, 1))
    sigma = zfit.Parameter("sigma", 0.5)
    lower = ((-5,),)
    upper = ((5,),)
    obs = zfit.Space(["obs1"], limits=(lower, upper))

    kde = zphys.unstable.pdf.GaussianKDE(data=data, bandwidth=sigma, obs=obs)

    probs = kde.pdf(x=data + 0.03)
    probs_np = zfit.run(probs)

    from zfit.core.loss import UnbinnedNLL
    data = zfit.Data.from_numpy(array=data, obs=obs)
    nll = UnbinnedNLL(model=kde, data=data)

    from zfit.minimizers.minimizer_minuit import Minuit
    minimizer = Minuit()

    minimum = minimizer.minimize(loss=nll)


# register the pdf here and provide sets of working parameter configurations

def _kde_params_factory():
    data = np.random.normal(size=(100, 3))
    sigmas = [0.5, 1., 2]
    lower = ((-5, -5, -5),)
    upper = ((5, 5, 5),)
    obs = zfit.Space(["obs1", "obs2", "obs3"], limits=(lower, upper))
    return {"data": data, "sigma": sigmas, "obs": obs}


tester.register_pdf(pdf_class=zphys.unstable.pdf.GaussianKDE, params_factories=_kde_params_factory())
