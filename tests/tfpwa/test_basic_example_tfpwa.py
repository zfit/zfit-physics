from pathlib import Path

import pytest
import zfit
from tf_pwa.config_loader import ConfigLoader

import zfit_physics.tfpwa as ztfpwa

this_dir = Path(__file__).parent


def generate_phsp_mc():
    """Take three-body decay A->BCD for example, we generate a PhaseSpace MC sample and a toy data sample."""

    datpath = (this_dir / "data")
    datpath.mkdir(exist_ok=True)

    print(f"Generate phase space MC: {datpath / 'PHSP.dat'}")
    generate_phspMC(Nmc=2000, mc_file=datpath / "PHSP.dat")
    print(f"Generate toy data: {datpath / 'data.dat'}")
    generate_toy_from_phspMC(Ndata=120, data_file=datpath / "data.dat")
    print("Done!")


def generate_phspMC(Nmc, mc_file):
    # We use ConfigLoader to read the information in the configuration file
    configpath = str(mc_file.parent.parent / "config.yml")
    config = ConfigLoader(configpath)
    # Set the parameters in the amplitude model
    config.set_params("gen_params.json")

    phsp = config.generate_phsp_p(Nmc)

    config.data.savetxt(str(mc_file), phsp)


def generate_toy_from_phspMC(Ndata, data_file):
    # We use ConfigLoader to read the information in the configuration file
    configpath = str(data_file.parent.parent / "config.yml")
    config = ConfigLoader(configpath)
    # Set the parameters in the amplitude model
    config.set_params("gen_params.json")

    data = config.generate_toy_p(Ndata)

    config.data.savetxt(str(data_file), data)
    return data


def test_example1_tfpwa():
    generate_phsp_mc()
    config = ConfigLoader(str(this_dir / "config.yml"))
    # Set init paramters. If not set, we will use random initial parameters
    config.set_params("gen_params.json")

    fcn = config.get_fcn()
    nll = ztfpwa.loss.nll_from_fcn(fcn)

    initial_val = config.get_fcn()(config.get_params())
    print(f"Initial value: {initial_val}")
    fit_result = config.fit(method="BFGS")

    # kwargs = dict(gradient=True, tol=0.01)
    print("initial NLL: ", nll.value())
    assert pytest.approx(nll.value(), 0.001) == initial_val
    kwargs = dict(tol=0.01)
    minimizer = zfit.minimize.Minuit(verbosity=0, **kwargs)
    # minimizer = zfit.minimize.NLoptMMAV1(verbosity=7, **kwargs)
    # minimizer = zfit.minimize.ScipyLBFGSBV1(verbosity=7, **kwargs)
    # minimizer = zfit.minimize.NLoptLBFGSV1(verbosity=7, **kwargs)
    print(f"Minimizer {minimizer} start with {kwargs}")
    result = minimizer.minimize(fcn)
    print(f"Finished minimization with config:{kwargs}")
    print(result)

    assert result.converged
    assert pytest.approx(result.fmin, 0.05) == fit_result.min_nll
