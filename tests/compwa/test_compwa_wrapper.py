from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import zfit_physics.compwa as zcompwa


# @pytest.fixture()
def create_amplitude():
    import qrules

    reaction = qrules.generate_transitions(
        initial_state=("J/psi(1S)", [-1, +1]),
        final_state=["gamma", "pi0", "pi0"],
        allowed_intermediate_particles=["f(0)"],
        allowed_interaction_types=["strong", "EM"],
        formalism="helicity",
    )

    import ampform
    from ampform.dynamics.builder import (
        create_non_dynamic_with_ff, create_relativistic_breit_wigner_with_ff)

    model_builder = ampform.get_builder(reaction)
    model_builder.scalar_initial_state_mass = True
    model_builder.stable_final_state_ids = [0, 1, 2]
    model_builder.set_dynamics("J/psi(1S)", create_non_dynamic_with_ff)
    for name in reaction.get_intermediate_particles().names:
        model_builder.set_dynamics(name, create_relativistic_breit_wigner_with_ff)
    model = model_builder.formulate()



    return model, reaction


def test_wrapper_simple_compwa():
    import zfit

    model, reaction = create_amplitude()

    from tensorwaves.function.sympy import create_parametrized_function

    unfolded_expression = model.expression.doit()
    intensity_func = create_parametrized_function(
        expression=unfolded_expression,
        parameters=model.parameter_defaults,
        backend="tensorflow",
    )

    from tensorwaves.data import SympyDataTransformer

    helicity_transformer = SympyDataTransformer.from_sympy(
        model.kinematic_variables, backend="numpy"
    )
    from tensorwaves.data import (IntensityDistributionGenerator,
                                  TFPhaseSpaceGenerator,
                                  TFUniformRealNumberGenerator,
                                  TFWeightedPhaseSpaceGenerator)

    rng = TFUniformRealNumberGenerator(seed=0)
    phsp_generator = TFPhaseSpaceGenerator(
        initial_state_mass=reaction.initial_state[-1].mass,
        final_state_masses={i: p.mass for i, p in reaction.final_state.items()},
    )
    phsp_momenta = phsp_generator.generate(100_000, rng)

    weighted_phsp_generator = TFWeightedPhaseSpaceGenerator(
        initial_state_mass=reaction.initial_state[-1].mass,
        final_state_masses={i: p.mass for i, p in reaction.final_state.items()},
    )
    data_generator = IntensityDistributionGenerator(
        domain_generator=weighted_phsp_generator,
        function=intensity_func,
        domain_transformer=helicity_transformer,
    )
    data_momenta = data_generator.generate(10_000, rng)

    phsp = helicity_transformer(phsp_momenta)
    data = helicity_transformer(data_momenta)
    data_frame = pd.DataFrame(data)
    phsp_frame = pd.DataFrame(phsp)

    initial_parameters = {
        R"C_{J/\psi(1S) \to {f_{0}(1500)}_{0} \gamma_{+1}; f_{0}(1500) \to \pi^{0}_{0} \pi^{0}_{0}}": (
            1.0
        ),
        "m_{f_{0}(500)}": 0.4,
        "m_{f_{0}(980)}": 0.88,
        "m_{f_{0}(1370)}": 1.22,
        "m_{f_{0}(1500)}": 1.45,
        "m_{f_{0}(1710)}": 1.83,
        R"\Gamma_{f_{0}(500)}": 0.3,
        R"\Gamma_{f_{0}(980)}": 0.1,
        R"\Gamma_{f_{0}(1710)}": 0.3,
    }

    free_parameter_symbols = [
        symbol
        for symbol in model.parameter_defaults
        if symbol.name in set(initial_parameters)
    ]  # TODO, use this?

    # TODO: cached doesn't really work, but needed?
    # cached_intensity_func, transform_to_cache = create_cached_function(
    #     unfolded_expression,
    #     parameters=model.parameter_defaults,
    #     free_parameters=free_parameter_symbols,
    #     backend="jax",
    # )
    # cached_data = transform_to_cache(data)
    # cached_phsp = transform_to_cache(phsp)

    # data conversion
    # phsp_zfit = zfit.Data.from_pandas(phsp_frame)
    # data_zfit = zfit.Data.from_pandas(data_frame)
    # data_frame = data_frame.astype(np.float64)
    # phsp_frame = phsp_frame.astype(np.float64)
    intensity = intensity_func

    pdf = zcompwa.pdf.ComPWAPDF(
        intensity=intensity,
        norm=pd.DataFrame(phsp).astype(np.float64),  # there are complex numbers in the norm
    )

    # pdf = zcompwa.pdf.ComPWAPDF(
    #     intensity=intensity,
    #     norm=phsp_frame,
    # )

    from tensorwaves.estimator import UnbinnedNLL

    estimator = UnbinnedNLL(
        intensity_func,
        data=data,
        phsp=phsp,
        backend="tensorflow",
    )


    loss = zfit.loss.UnbinnedNLL(pdf, data_frame, options={'numgrad': True})

    # cannot convert, cannot compare to the ComPWA gradient as it's not available or erros
    # np.testing.assert_allclose(loss.gradient(), estimator.gradient(initial_parameters), rtol=1e-5)

    minimizer = zfit.minimize.Minuit(verbosity=7, gradient=True)
    # minimizer = zfit.minimize.Minuit(verbosity=7, gradient='zfit')
    # minimizer = zfit.minimize.ScipyLBFGSBV1(verbosity=8)
    # minimizer = zfit.minimize.ScipyBFGS(verbosity=9)
    # minimizer = zfit.minimize.ScipyTrustKrylovV1(verbosity=8)
    # minimizer = zfit.minimize.NLoptMMAV1(verbosity=9)
    # minimizer = zfit.minimize.IpyoptV1(verbosity=8)
    params = loss.get_params()
    paramsfit = [p for p in params
                 if p.name in initial_parameters
                 or p.name.endswith('_REALPART') and p.name[:-9] in initial_parameters  # if complex, parts are indep
                 or p.name.endswith('_IMAGPART') and p.name[:-9] in initial_parameters]
    nll_estimator = zcompwa.loss.nll_from_estimator(estimator, numgrad=True)
    _ = nll_estimator.value()
    # TODO: works but is slow
    gradient_est = nll_estimator.gradient(list(nll_estimator.get_params())[:2])
    assert not any(np.isnan(gradient_est))
    gradient_zfit = loss.gradient(list(loss.get_params())[:2])
    assert not any(np.isnan(gradient_zfit))
    # np.testing.assert_allclose(gradient_est, gradient_zfit, rtol=1e-5)

    from tensorwaves.optimizer import Minuit2

    minuit2 = Minuit2(
        use_analytic_gradient=False,
    )
    fit_result = minuit2.optimize(estimator, initial_parameters)
    # print(fit_result)

    with zfit.param.set_values(params, params):
        result = minimizer.minimize(loss, params=paramsfit)
    # print(result)
    # TODO: test values? But ComPWA has bad values
    # for p in paramsfit:
    #     if p.name.endswith('_REALPART') or p.name.endswith('_IMAGPART'):
    #         continue
    #     if p.name not in initial_parameters:
    #         print(f'Not in initial, ERROR: {p.name}')
    #         continue
    #     comp = fit_result.parameter_values[p.name]
    #     print(f"{p.name}, diff {p - comp}: {p.numpy()}, {comp}")
    result.hesse()
    # print(result)
    assert result.valid
    tol = 0.05  # 10% of 1 sigma
    assert result.fmin - tol < fit_result.estimator_value  # ComPWA doesn't minimize well, if this fails, we can relax it
    assert pytest.approx(result.fmin, abs=0.5) == fit_result.estimator_value
