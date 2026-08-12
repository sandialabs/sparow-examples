import time
from sparow.sp import stochastic_program
from sparow.ef import ExtensiveFormSolver
from sparow.ph import ProgressiveHedgingSolver

def run_ef(model_data_dict, builder_dict, first_stage_variables, bundle_config=None, solver_name="ipopt"):
    sp = stochastic_program(first_stage_variables=first_stage_variables)

    for name, model_data in model_data_dict.items():
        sp.initialize_model(name=name, model_data=model_data, model_builder=builder_dict[name])

    if bundle_config is not None:
        sp.initialize_bundles(**bundle_config)

    solver = ExtensiveFormSolver()
    solver.set_options(solver=solver_name, loglevel="INFO")

    start = time.time()
    results = solver.solve(sp)
    end = time.time()

    return {
        "sp": sp,
        "results": results,
        "cpu_time": end - start,
    }

def run_ph(model_data_dict, builder_dict, first_stage_variables, max_iterations, default_rho, bundle_config=None, solver_name="ipopt"):
    sp = stochastic_program(first_stage_variables=first_stage_variables)

    for name, model_data in model_data_dict.items():
        sp.initialize_model(name=name, model_data=model_data, model_builder=builder_dict[name])

    if bundle_config is not None:
        sp.initialize_bundles(**bundle_config)

    solver = ProgressiveHedgingSolver()
    solver.set_options(
        solver=solver_name,
        loglevel="INFO",
        max_iterations=max_iterations,
        default_rho=default_rho,
    )

    start = time.time()
    results = solver.solve(sp)
    end = time.time()

    return {
        "sp": sp,
        "results": results,
        "cpu_time": end - start,
    }
