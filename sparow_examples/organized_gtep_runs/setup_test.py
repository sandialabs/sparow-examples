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
    representative_dates: dict[str, list[str]],
    representative_weights: dict[str, list[float]],
    power_flow_fidelity: dict,
    relaxations: dict,
    include_commitment: dict,
    number_of_commitment: dict,
    base_dir: str = ".",
) -> None:
    """
    Create a master experiment directory and per-scenario subdirectories.
    """

    scenario_dicts = {
        "alpha": alpha,
        "num_representative_days": num_representative_days,
        "representative_dates": representative_dates,
        "representative_weights": representative_weights,
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

    for scenario in scenarios:
        n_rep = num_representative_days[scenario]
        dates = representative_dates[scenario]
        weights = representative_weights[scenario]

        if len(dates) != n_rep:
            raise ValueError(
                f"Scenario '{scenario}' has num_representative_days={n_rep}, "
                f"but {len(dates)} representative dates were provided."
            )

        if len(weights) != n_rep:
            raise ValueError(
                f"Scenario '{scenario}' has num_representative_days={n_rep}, "
                f"but {len(weights)} representative weights were provided."
            )

    experiment_path = Path(base_dir) / experimental_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    master_config = {
        "experimental_name": experimental_name,
        "case_study": case_study,
        "scenarios": scenarios,
        "growth_rate": growth_rate,
        "num_representative_days": num_representative_days,
        "representative_dates": representative_dates,
        "representative_weights": representative_weights,
    }

    with open(experiment_path / "experiment_config.json", "w") as f:
        json.dump(master_config, f, indent=4)

    execute_run_template = f'''import pytest
from sparow.ef import ExtensiveFormSolver
from sparow.ph import ProgressiveHedgingSolver
import pyomo.opt
from pyomo.common import unittest
from sparow.sp.util import relax_second_stage
import time
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from post_process_gtep_solution import post_process

solvers = set(pyomo.opt.check_available_solvers("gurobi"))

try:
    from sparow_examples.organized_gtep_runs.{experimental_name} import create_sp

    dummy_available = True
except:
    dummy_available = False

relaxations = {repr(relaxations)}
rd = {{k: v["relax_second_stage"] for k, v in relaxations.items()}}

script_start = time.perf_counter()
sp = create_sp()
sp.add_transformation(relax_second_stage, relax_dict=rd)
solver = ExtensiveFormSolver()
solver.set_options(solver="gurobi", loglevel="INFO")

solve_start = time.perf_counter()
res_munch = solver.solve_and_return_EF(sp)
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
post_process(mod_object)
'''
    with open(experiment_path / "execute_run.py", "w") as f:
        f.write(execute_run_template)

    source_model_dir = Path.cwd() / "new_model"
    if not source_model_dir.exists() or not source_model_dir.is_dir():
        raise FileNotFoundError(
            f"Could not find source data directory at: {source_model_dir}"
        )

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
                "representative_dates": representative_dates[scenario],
                "representative_weights": representative_weights[scenario],
                "num_commit": number_of_commitment[scenario],
                "include_commitment": include_commitment[scenario],
            }
        )

    app_data = {
        "stages": 3,
        "len_reps": 24,
        "num_dispatch": 1,
    }

    experiment_init_content = f'''# sparow_examples.organized_gtep_runs.{experimental_name}

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
    representative_dates = data["representative_dates"]
    representative_weights = data["representative_weights"]

    scenario = importlib.import_module(
        "sparow_examples.organized_gtep_runs.{experimental_name}." + data["ID"]
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
        representative_dates=representative_dates,
        representative_weights=representative_weights,
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
            "representative_dates": representative_dates[scenario],
            "representative_weights": representative_weights[scenario],
            "power_flow_fidelity": power_flow_fidelity[scenario],
            "relaxations": relaxations[scenario],
            "number_of_commitment": number_of_commitment[scenario],
            "include_commitment": include_commitment[scenario],
        }

        destination_model_dir = scenario_path
        if destination_model_dir.exists():
            shutil.rmtree(destination_model_dir)
        shutil.copytree(source_model_dir, destination_model_dir)

        with open(scenario_path / "scenario_config.json", "w") as f:
            json.dump(scenario_config, f, indent=4)

    print(f"Experimental setup created at: {experiment_path.resolve()}")


if __name__ == "__main__":

    experimental_name = "test_gtep"
    case_study = "9-bus"

    scenarios = ["scenario_A"]

    alpha = {
        "scenario_A": 1.0,
    }

    growth_rate = 1.00

    num_representative_days = {
        "scenario_A": 4,
    }

    power_flow_fidelity = {
        "scenario_A": "DC",
    }

    relaxations = {
        "scenario_A": {"relax_second_stage": False, "unit_commitment": True},
    }

    include_commitment = {
        "scenario_A": True
    }

    number_of_commitment = {
        "scenario_A": 24,
    }

    representative_dates = {
        "scenario_A": [
            "2020-01-28 00:00",
            "2020-04-23 00:00",
            "2020-07-05 00:00",
            "2020-10-14 00:00",
        ],
    }

    representative_weights = {
        "scenario_A": [1, 1, 1, 1],
    }

    create_experimental_setups(
    experimental_name=experimental_name,
    case_study=case_study,
    scenarios=scenarios,
    alpha=alpha,
    growth_rate=growth_rate,
    num_representative_days=num_representative_days,
    representative_dates=representative_dates,
    representative_weights=representative_weights,
    power_flow_fidelity=power_flow_fidelity,
    relaxations=relaxations,
    number_of_commitment=number_of_commitment,
    include_commitment=include_commitment,
    base_dir="."
)
