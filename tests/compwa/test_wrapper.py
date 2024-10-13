from __future__ import annotations

import numpy as np
import pandas as pd

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

    from tensorwaves.data import (TFPhaseSpaceGenerator,
                                  TFUniformRealNumberGenerator)

    rng = TFUniformRealNumberGenerator(seed=0)
    phsp_generator = TFPhaseSpaceGenerator(
        initial_state_mass=reaction.initial_state[-1].mass,
        final_state_masses={i: p.mass for i, p in reaction.final_state.items()},
    )
    phsp_momenta = phsp_generator.generate(100_000, rng)

    unfolded_expression = model.expression.doit()

    return model, reaction


def test_wrapper_simple():
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

    # data conversion
    # phsp_zfit = zfit.Data.from_pandas(phsp_frame)
    # data_zfit = zfit.Data.from_pandas(data_frame)
    data_frame = data_frame.astype(np.float64)
    phsp_frame = phsp_frame.astype(np.float64)
    intensity = intensity_func

    pdf = zcompwa.pdf.ComPWAPDF(
        intensity=intensity,
        norm=phsp_frame,
    )
    # for p in pdf.get_params():  # returns the free params automatically
    #     p.set_value(p + np.random.normal(0, 0.01))
    # zfit.run.set_graph_mode(False)
    zfit.run.set_autograd_mode(False)
    loss = zfit.loss.UnbinnedNLL(pdf, data_frame)

    # ok, here I was caught up playing around :) Minuit seems to perform the best though
    minimizer = zfit.minimize.Minuit(verbosity=7, gradient=True)
    # minimizer = zfit.minimize.Minuit(verbosity=7, gradient='zfit')
    # minimizer = zfit.minimize.ScipyLBFGSBV1(verbosity=8)
    # minimizer = zfit.minimize.ScipyTrustKrylovV1(verbosity=8)
    # minimizer = zfit.minimize.NLoptMMAV1(verbosity=9)
    # minimizer = zfit.minimize.IpyoptV1(verbosity=8)
    result = minimizer.minimize(loss)
    print(result)
    result.hesse()
    print(result)
    assert result.valid
