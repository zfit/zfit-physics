"""Example test for a pdf or function"""

import zfit
import numpy as np
# Important, do the imports below
from zfit.core.testing import setup_function, teardown_function, tester

import zfit_physics as zphys

# specify globals here. Do NOT add any TensorFlow but just pure python
param1_true = 0.3
param2_true = 1.2


def test_simple_kde():
    # test special properties  here
    data = np.random.normal(size=(100, 3))
    sigmas = [0.5, 1., 2]
    lower = ((-5, -5, -5),)
    upper = ((5, 5, 5),)
    obs = zfit.Space(["obs1", "obs2", "obs3"], limits=(lower, upper))

    kde = zphys.pdf.GaussianKDE(data=data, sigma=sigmas, obs=obs)

    probs = kde.pdf(x=data + 0.03)
    probs_np = zfit.run(probs)


# register the pdf here and provide sets of working parameter configurations

def _kde_params_factory():
    data = np.random.normal(size=(100, 3))
    sigmas = [0.5, 1., 2]
    lower = ((-5, -5, -5),)
    upper = ((5, 5, 5),)
    obs = zfit.Space(["obs1", "obs2", "obs3"], limits=(lower, upper))
    return {"data": data, "sigma": sigmas, "obs": obs}


tester.register_pdf(pdf_class=zphys.pdf.GaussianKDE, params_factories=_kde_params_factory())
