from pathlib import Path
import json
import shutil


def create_experimental_setups(
    experimental_name: str,
    case_study: str,
    scenarios: list[str],
    alpha: dict,
    growth_rate: float,
    num_representative_days: dict[str, int],
    power_flow_fidelity: dict,
    relaxations: dict,
    include_commitment: dict,
    number_of_commitment: dict,
    base_dir: str = ".",
) -> None:
    """
    Create a master experiment directory and per-scenario subdirectories.

    Each directory gets an __init__.py.
    Each scenario directory also gets:
      - scenario_config.json
      - driver_gtep.py
      - data/ copied from ./data in the current working directory
    """

    scenario_dicts = {
        "alpha": alpha,
        "num_representative_days": num_representative_days,
        "power_flow_fidelity": power_flow_fidelity,
        "relaxations": relaxations,
        "include_commitment": include_commitment,
        "number_of_commitment": number_of_commitment,
    }

    for dict_name, d in scenario_dicts.items():
        missing = [s for s in scenarios if s not in d]
        if missing:
            raise ValueError(
                f"Missing scenario entries in '{dict_name}' for: {missing}"
            )

    experiment_path = Path(base_dir) / experimental_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    # Save master config
    master_config = {
        "experimental_name": experimental_name,
        "case_study": case_study,
        "scenarios": scenarios,
        "growth_rate": growth_rate,
        "num_representative_days": num_representative_days,
    }

    with open(experiment_path / "experiment_config.json", "w") as f:
        json.dump(master_config, f, indent=4)

    execute_run_template = f'''import pytest
from sparow.ef import ExtensiveFormSolver
from sparow.ph import ProgressiveHedgingSolver
from sparow.benders import BendersSolver
import pyomo.opt
from pyomo.common import unittest
from sparow.sp.util import relax_second_stage
import time
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#from post_process_gtep_solution import post_process

solvers = set(pyomo.opt.check_available_solvers("gurobi"))

try:
    from sparow_examples.bus_123_benders.{experimental_name} import create_sp

    dummy_available = True
except:
    dummy_available = False


relaxations = {repr(relaxations)}
rd = {{k: v["relax_second_stage"] for k, v in relaxations.items()}}

script_start = time.perf_counter()
sp = create_sp()
sp.add_transformation(relax_second_stage, relax_dict=rd)
solver = BendersSolver()

TIME_LIMIT = 1200.0
CONVERGENCE_TOL = 1e-3
MAX_ITERATIONS = 100
ETA_LOWER_BOUND_DEFAULT = -1e6

eta_bounds_map = {{b: (ETA_LOWER_BOUND_DEFAULT, None) for b in sp.bundles}}


def _on_iteration(data):
    n_cuts = len(data.cuts_added) if data.cuts_added is not None else 0
    print(
        f"  --- Benders iteration {{data.iter_idx}}  cuts_added={{n_cuts}} ---",
        flush=True,
    )

solver.set_options(
    solver="gurobi_persistent",
    subproblem_solver="gurobi_persistent",
    max_iterations=MAX_ITERATIONS,
    is_persistent_solver=True,
    allow_infeasible_subproblems=True,
    loglevel="INFO",
    solver_options={{"TimeLimit": TIME_LIMIT, "timelimit": TIME_LIMIT}},
)

print(
    f"--- Calling solve_and_return_model "
    f"(TimeLimit={{TIME_LIMIT}}s, convergence_tol={{CONVERGENCE_TOL}}) ---",
    flush=True,
)

solve_start = time.perf_counter()
res_munch = solver.solve_and_return_model(
    sp,
    eta_bounds_map,
    #subproblem_transforms=[relax_second_stage],
    master_transforms=None,
    on_iteration=_on_iteration,
    convergence_tol=CONVERGENCE_TOL,
)
results = res_munch.solutions
solve_end = time.perf_counter()

results_dict = results.to_dict()

soln = next(iter(results_dict["solutions"].values()))

obj_val = soln["objectives"][0]["value"]
total_end = time.perf_counter()
print(f"Objective value: {{obj_val}}")
print(f"Solve time (seconds): {{solve_end - solve_start:.4f}}")
print(f"Total runtime (seconds): {{total_end - script_start:.4f}}")

mod_object=res_munch.model.s['model','scenario_A']
#post_process(mod_object)
'''
    with open(experiment_path / "execute_run.py", "w") as f:
        f.write(execute_run_template)

    source_model_dir = Path.cwd() / "model"
    if not source_model_dir.exists() or not source_model_dir.is_dir():
        raise FileNotFoundError(
            f"Could not find source data directory at: {source_model_dir}"
        )

    # Build model_data["scenarios"] dynamically
    model_scenarios = []
    probability = 1.0 / len(scenarios) if scenarios else 0.0

    for scenario in scenarios:
        model_scenarios.append(
            {
                "ID": scenario,
                "Demand": growth_rate,
                "Probability": probability,
                "alpha": alpha[scenario],
                "PF": power_flow_fidelity[scenario],
                "num_reps": num_representative_days[scenario],
                "num_commit": number_of_commitment[scenario],
                "include_commitment": include_commitment[scenario],
            }
        )

    # Build experiment-level __init__.py
    app_data = {
        "stages": 3,
        #"num_reps": num_representative_days,
        "len_reps": 24,
       # "num_commit": next(iter(number_of_commitment.values())),
        "num_dispatch": 1,
    }

    experiment_init_content = f'''# sparow_examples.bus_123_benders.{experimental_name}

from sparow.sp import stochastic_program
import importlib


app_data = {repr(app_data)}
model_data = {repr({"scenarios": model_scenarios})}


def model_builder(data, args):
    num_stages = data["stages"]
    num_rep_days = data["num_reps"]
    len_rep_days = data["len_reps"]
    num_commit_p = data["num_commit"]
    num_disp = data["num_dispatch"]
    alpha = data["alpha"]
    PF = data["PF"]
    include_commitment = data["include_commitment"]

    scenario = importlib.import_module(
        "sparow_examples.bus_123_benders.{experimental_name}." + data["ID"]
    )
    return scenario.create_gtep_model(
        num_stages=num_stages,
        num_rep_days=num_rep_days,
        len_rep_days=len_rep_days,
        num_commit_p=num_commit_p,
        num_disp=num_disp,
        alpha=alpha,
        flow_model=PF,
        include_commitment=include_commitment,
    )


def create_sp():
    sp = stochastic_program(
        first_stage_variables=[
            "investmentStage[*].renewableOperational[*]",
            "investmentStage[*].renewableInstalled[*]",
            "investmentStage[*].renewableRetired[*]",
            "investmentStage[*].renewableExtended[*]",
            "investmentStage[*].renewableDisabled[*]",
            "investmentStage[*].genOperational[*].binary_indicator_var",
            "investmentStage[*].genInstalled[*].binary_indicator_var",
            "investmentStage[*].genRetired[*].binary_indicator_var",
            "investmentStage[*].genDisabled[*].binary_indicator_var",
            "investmentStage[*].genExtended[*].binary_indicator_var",
            "investmentStage[*].branchOperational[*].binary_indicator_var",
            "investmentStage[*].branchInstalled[*].binary_indicator_var",
            "investmentStage[*].branchRetired[*].binary_indicator_var",
            "investmentStage[*].branchDisabled[*].binary_indicator_var",
            "investmentStage[*].branchExtended[*].binary_indicator_var",
        ]
    )
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="model", model_data=model_data, model_builder=model_builder
    )
    return sp
'''
    with open(experiment_path / "__init__.py", "w") as f:
        f.write(experiment_init_content)

    # Create per-scenario directories and config files
    for scenario in scenarios:
        scenario_path = experiment_path / scenario
        scenario_path.mkdir(parents=True, exist_ok=True)

        scenario_config = {
            "experimental_name": experimental_name,
            "case_study": case_study,
            "scenario": scenario,
            "alpha": alpha[scenario],
            "growth_rate": growth_rate,
            "num_representative_days": num_representative_days[scenario],
            "power_flow_fidelity": power_flow_fidelity[scenario],
            "relaxations": relaxations[scenario],
            "number_of_commitment": number_of_commitment[scenario],
            "include_commitment": include_commitment[scenario],
        }

        # Copy data directory into scenario directory
        destination_model_dir = scenario_path
        if destination_model_dir.exists():
            shutil.rmtree(destination_model_dir)
        shutil.copytree(source_model_dir, destination_model_dir)

        with open(scenario_path / "scenario_config.json", "w") as f:
            json.dump(scenario_config, f, indent=4)

    print(f"Experimental setup created at: {experiment_path.resolve()}")


if __name__ == "__main__":

    experimental_name = "three_alpha_scenarios"
    case_study = "bus_123_benders"

    scenarios = ["scenario_A", "scenario_B", "scenario_C"]

    alpha = {
        "scenario_A": 1.0,"scenario_B": 1.0,"scenario_C": 1.0,
    }

    growth_rate = 1.00

    num_representative_days = {
        "scenario_A": 1,"scenario_B": 1,"scenario_C": 1
    }

    power_flow_fidelity = {
        "scenario_A": "CP","scenario_B": "CP","scenario_C": "CP"
    }

    relaxations = {
        "scenario_A": {"relax_second_stage": True, "unit_commitment": True},
        "scenario_B": {"relax_second_stage": True, "unit_commitment": True},
        "scenario_C": {"relax_second_stage": True, "unit_commitment": True},
    }

    include_commitment = {
        "scenario_A": True,
        "scenario_B": True,
        "scenario_C": True,
    }

    number_of_commitment = {
        "scenario_A": 1,
        "scenario_B": 1,
        "scenario_C": 1,
    }

    create_experimental_setups(
        experimental_name=experimental_name,
        case_study=case_study,
        scenarios=scenarios,
        alpha=alpha,
        growth_rate=growth_rate,
        num_representative_days=num_representative_days,
        power_flow_fidelity=power_flow_fidelity,
        relaxations=relaxations,
        number_of_commitment=number_of_commitment,
        include_commitment=include_commitment,
        base_dir="."
    )
