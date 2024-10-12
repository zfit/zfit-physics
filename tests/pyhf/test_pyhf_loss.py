import numpy as np
import pyhf
import zfit
from pyhf.simplemodels import uncorrelated_background

import zfit_physics.pyhf as zpyhf


def generate_source_static(n_bins):
    """Create the source structure for the given number of bins.

    Args:
        n_bins: `list` of number of bins

    Returns:
        source
    """
    scale = 10
    binning = [n_bins, -0.5, n_bins + 0.5]
    data = np.random.poisson(size=n_bins, lam=scale * 120).tolist()
    bkg = np.random.poisson(size=n_bins, lam=scale * 100).tolist()
    bkgerr = np.random.normal(size=n_bins, loc=scale * 10.0, scale=3).tolist()
    sig = np.random.poisson(size=n_bins, lam=scale * 30).tolist()

    source = {
        "binning": binning,
        "bindata": {"data": data, "bkg": bkg, "bkgerr": bkgerr, "sig": sig},
    }
    return source


def test_nll_from_pyhf_simple():
    nbins = 50
    source = generate_source_static(nbins)

    signp = source["bindata"]["sig"]
    bkgnp = source["bindata"]["bkg"]
    uncnp = source["bindata"]["bkgerr"]
    datanp = source["bindata"]["data"]

    pdf = uncorrelated_background(signp, bkgnp, uncnp)
    data = datanp + pdf.config.auxdata

    nll = zpyhf.loss.nll_from_pyhf(data, pdf)

    minimizer = zfit.minimize.Minuit(verbosity=7)
    resultz = minimizer.minimize(nll)

    values, fmin = pyhf.infer.mle.fit(
        data, pdf, pdf.config.suggested_init(), pdf.config.suggested_bounds(), return_fitted_val=True
    )
    assert np.allclose(resultz.fmin, fmin / 2, atol=1e-4)
    np.testing.assert_allclose(resultz.values, values, atol = 5e-3)
