import numpy as np
import pytest

_ = pytest.importorskip("ROOT")


def test_loss_registry():
    import zfit

    import zfit_physics.roofit as zroofit

    # create space
    obs = zfit.Space("x", -2, 3)

    # parameters
    mu = zfit.Parameter("mu", 1.2, -4, 6)
    sigma = zfit.Parameter("sigma", 1.3, 0.5, 10)

    # model building, pdf creation
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    # data
    ndraw = 10_000
    data = np.random.normal(loc=2.0, scale=3.0, size=ndraw)
    data = obs.filter(data)  # works also for pandas DataFrame

    from ROOT import RooArgSet, RooDataSet, RooGaussian, RooRealVar

    mur = RooRealVar("mu", "mu", 1.2, -4, 6)
    sigmar = RooRealVar("sigma", "sigma", 1.3, 0.5, 10)
    obsr = RooRealVar("x", "x", -2, 3)
    gaussr = RooGaussian("gauss", "gauss", obsr, mur, sigmar)

    datar = RooDataSet("data", "data", {obsr})
    for d in data:
        obsr.setVal(d)
        datar.add(RooArgSet(obsr))

    # create a loss function
    nll = gaussr.createNLL(datar)
    nll_fromroofit = zroofit.loss.nll_from_roofit(nll)

    nllz = zfit.loss.UnbinnedNLL(model=gauss, data=data)

    # create a minimizer
    tol = 1e-3
    verbosity = 0
    minimizer = zfit.minimize.Minuit(gradient=True, verbosity=verbosity, tol=tol, mode=1)
    minimizerzgrad = zfit.minimize.Minuit(gradient=False, verbosity=verbosity, tol=tol, mode=1)

    params = nllz.get_params()
    initvals = np.array(params)

    with zfit.param.set_values(params, initvals):
        result = minimizer.minimize(nllz)

    with zfit.param.set_values(params, initvals):
        result2 = minimizer.minimize(nll)

    assert result.params['mu']['value'] == pytest.approx(result2.params['mu']['value'], rel=1e-3)
    assert result.params['sigma']['value'] == pytest.approx(result2.params['sigma']['value'], rel=1e-3)

    with zfit.param.set_values(params, params):
        result4 = minimizerzgrad.minimize(nll)

    assert result.params['mu']['value'] == pytest.approx(result4.params['mu']['value'], rel=1e-3)
    assert result.params['sigma']['value'] == pytest.approx(result4.params['sigma']['value'], rel=1e-3)

    with zfit.param.set_values(params, params):
        result5 = minimizerzgrad.minimize(nll_fromroofit)
