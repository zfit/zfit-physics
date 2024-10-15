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
    # generate_phsp_mc()
    config = ConfigLoader(str(this_dir / "config.yml"))
    # Set init paramters. If not set, we will use random initial parameters
    config.set_params("gen_params.json")

    # fit_result = config.fit(method="L-BFGS-B")
    # print(f"Fit parameters: {fit_result}")
    # calculate Hesse errors of the parameters
    # errors = config.get_params_error(fit_result)

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

    #
    # if data is None and phsp is None:
    #     data, phsp, bg, inmc = self.get_all_data()
    #     fcn = self.get_fcn(batch=batch)
    # else:
    #     fcn = self.get_fcn([data, phsp, bg, inmc], batch=batch)
    # if self.config["data"].get("lazy_call", False):
    #     print_init_nll = False
    #     # print("sss")
    # amp = self.get_amplitude()
    # print("decay chains included: ")
    # for i in self.full_decay:
    #     ls_list = [getattr(j, "get_ls_list", lambda x: None)() for j in i]
    #     print("  ", i, " ls: ", *ls_list)
    # if reweight:
    #     ConfigLoader.reweight_init_value(
    #         amp, phsp[0], ns=data_shape(data[0])
    #     )
    #
    # print("\n########### initial parameters")
    # print(json.dumps(amp.get_params(), indent=2), flush=True)
    # if print_init_nll:
    #     print("initial NLL: ", fcn({}))  # amp.get_params()))
    # fit configure
    # self.bound_dic[""] = (,)

    # start = time.time()
    #
    # import zfit
    # minimizer = zfit.minimize.Minuit(verbosity=7, tol=0.2, gradient=False)
    # # minimizer = zfit.minimize.IpyoptV1(verbosity=7, tol=0.2)
    # # minimizer = zfit.minimize.ScipyLBFGSBV1(verbosity=5, tol=0.2, gradient='zfit')
    # # minimizer = zfit.minimize.ScipyTrustConstrV1(verbosity=5, tol=0.2, gradient='zfit')
    # # minimizer = zfit.minimize.NLoptTruncNewtonV1(verbosity=5, tol=0.2)
    # # minimizer = zfit.minimize.NLoptLBFGSV1(verbosity=5, tol=0.1)
    # # minimizer= zfit.minimize.NLoptStoGOV1(verbosity=5, tol=0.01)
    # # minimizer = zfit.minimize.ScipyNewtonCGV1(verbosity=5, tol=0.2, gradient='zfit')
    # # minimizer = zfit.minimize.ScipyPowellV1(verbosity=5, tol=0.2)
    # # minimizer = zfit.minimize.NLoptCCSAQV1(verbosity=5, tol=0.2)
    # # min
    #
    #
    # zfit_params = [zfit.Parameter(n, v, floating=n in fcn.vm.trainable_vars) for n, v in amp.get_params().items()]
    # class TFPWALoss(zfit.loss.BaseLoss):
    #     def __init__(self, loss, params=None):
    #         if params is None:
    #             params =  [zfit.Parameter(n, v) for n, v in amp.get_params().items() if n in fcn.vm.trainable_vars]
    #         self._lossparams = params
    #         super().__init__(model=[], data=[], options={"subtr_const": False}, jit=False)
    #         self._errordef = 0.5
    #         self._tfpwa_loss = loss
    #
    #     def _value(self, model, data, fit_range, constraints, log_offset):
    #         return self._tfpwa_loss(self._lossparams)
    #
    #     def _value_gradient(self, params, numgrad, full=None):
    #         return self._tfpwa_loss.get_nll_grad(params)
    #
    #     def _value_gradient_hessian(self, params, hessian, numerical=False, full: bool | None = None):
    #         return self._tfpwa_loss.get_nll_grad_hessian(params)
    #
    #     # below is a small hack as zfit is reworking it's loss currently
    #     def _get_params(
    #             self,
    #             floating: bool | None = True,
    #             is_yield: bool | None = None,
    #             extract_independent: bool | None = True,
    #     ):
    #         params = super()._get_params(floating, is_yield, extract_independent)
    #         from zfit.core.baseobject import extract_filter_params
    #         own_params = extract_filter_params(self._lossparams, floating=floating, extract_independent=extract_independent)
    #         return params.union(own_params)
    #
    #     def create_new(self):
    #         raise RuntimeError("Not needed, todo")
    #
    #     def _loss_func(self,):
    #         raise RuntimeError("Not needed, needs new release")
    #
    # # loss = TFPWALoss(fcn, zfit_params)
    # loss = zfit.loss.SimpleLoss(func=lambda x: fcn(x), params=zfit_params, errordef=0.5, gradient=lambda x: fcn.get_nll_grad(x)[1], hessian=lambda x: fcn.get_nll_grad_hessian(x)[2])
    # result = minimizer.minimize(loss, zfit_params)
    # print(result)
    # print(f"minimize time: {time.time() - start}")
    # print(f"Number of evaluations: {result.info}")
    # # starthesse = time.time()
    # # result.hesse()
    # # print(f"hesse time: {time.time() - starthesse}")
    # # starthessenp = time.time()
    # # print(f"hesse: {loss.hessian()}")
    # # result.hesse(method="hesse_np", name="hesse_np")
    # # print(f"hesse_np time: {time.time() - starthessenp}")
    # # startminos = time.time()
    # # result.errors()
    # # print(f"minos time: {time.time() - startminos}")
    # # starterrors = time.time()
    # # try:
    # #     result.errors(method="zfit_errors", name="zfit_errors")
    # # except Exception as error:
    # #     print(f"errors failed with {error}")
    # # print(f"errors time: {time.time() - starterrors}")
    # print(result)
    # print("total fit time: ", time.time() - start)
    # # exit()
    #
    # self.fit_params = fit(
    #     use="minuit",
    #     fcn=fcn,
    #     method=method,
    #     bounds_dict=self.bound_dic,
    #     check_grad=check_grad,
    #     improve=False,
    #     maxiter=maxiter,
    #     jac=jac,
    #     callback=callback,
    #     grad_scale=grad_scale,
    #     gtol=gtol,
    # )
    # print(f"Fit parameters: {self.fit_params}")
    # print("fit time: ", time.time() - start)
