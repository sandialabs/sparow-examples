from pathlib import Path
import json
import shutil


def create_experimental_setups(
    experimental_name: str,
    case_study: str,
    scenarios: list[str],
    alpha: dict,
    growth_rate: float,
    num_representative_days: int,
    power_flow_fidelity: dict,
    relaxations: dict,
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
        "power_flow_fidelity": power_flow_fidelity,
        "relaxations": relaxations,
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
import pyomo.opt
from pyomo.common import unittest
from sparow.sp.util import relax_second_stage

solvers = set(pyomo.opt.check_available_solvers("gurobi"))

try:
    from sparow_examples.organized_gtep_runs.{experimental_name} import create_sp

    dummy_available = True
except:
    dummy_available = False

relaxations = {repr(relaxations)}
rd = {{k: v["relax_second_stage"] for k, v in relaxations.items()}}
sp = create_sp()
sp.add_transformation(relax_second_stage, relax_dict=rd)
solver = ExtensiveFormSolver()
solver.set_options(solver="gurobi", loglevel="INFO")
results = solver.solve(sp)
results_dict = results.to_dict()

soln = next(iter(results_dict["solutions"].values()))

obj_val = soln["objectives"][0]["value"]
print(obj_val)
'''
    with open(experiment_path / "execute_run.py", "w") as f:
        f.write(execute_run_template)

    source_data_dir = Path.cwd() / "data"
    if not source_data_dir.exists() or not source_data_dir.is_dir():
        raise FileNotFoundError(
            f"Could not find source data directory at: {source_data_dir}"
        )

    # Template driver file contents
    driver_template = '''from pathlib import Path
from pyomo.core import TransformationFactory
from gtep.gtep_model import ExpansionPlanningModel
from gtep.gtep_data import ExpansionPlanningData

current_file_dir = Path(__file__).resolve().parent


def create_gtep_model(
    *, num_stages, num_rep_days, len_rep_days, num_commit_p, num_disp, alpha=1.0, flow_model="CP"
):
    data_path = str(current_file_dir / "data")
    data_object = ExpansionPlanningData()
    data_object.load_prescient(data_path)
    # data_object.load_storage_csv(data_path)

    mod_object = ExpansionPlanningModel(
        stages=num_stages,
        data=data_object,
        num_reps=num_rep_days,
        len_reps=len_rep_days,
        num_commit=num_commit_p,
        num_dispatch=num_disp,
    )

    mod_object.config["include_commitment"] = True
    mod_object.config["alpha_scaler"] = alpha
    mod_object.config["flow_model"] = flow_model
    mod_object.config["storage"] = True
    mod_object.config["transmission"] = True
    mod_object.config["thermal_generation"] = True
    mod_object.config["renewable_generation"] = True
    mod_object.config["scale_loads"] = False
    mod_object.config["scale_texas_loads"] = False

    mod_object.create_model()
    TransformationFactory("gdp.bound_pretransformation").apply_to(mod_object.model)
    TransformationFactory("gdp.bigm").apply_to(mod_object.model)

    return mod_object.model
'''

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
            }
        )

    # Build experiment-level __init__.py
    app_data = {
        "stages": 3,
        "num_reps": num_representative_days,
        "len_reps": 1,
        "num_commit": next(iter(number_of_commitment.values())),
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
            "num_representative_days": num_representative_days,
            "power_flow_fidelity": power_flow_fidelity[scenario],
            "relaxations": relaxations[scenario],
            "number_of_commitment": number_of_commitment[scenario],
        }

        with open(scenario_path / "scenario_config.json", "w") as f:
            json.dump(scenario_config, f, indent=4)

        # Create scenario __init__.py
        with open(scenario_path / "__init__.py", "w") as f:
            f.write("from .driver_gtep import create_gtep_model\n")

        # Create driver_gtep.py
        with open(scenario_path / "driver_gtep.py", "w") as f:
            f.write(driver_template)

        # Copy data directory into scenario directory
        destination_data_dir = scenario_path / "data"
        if destination_data_dir.exists():
            shutil.rmtree(destination_data_dir)
        shutil.copytree(source_data_dir, destination_data_dir)

    print(f"Experimental setup created at: {experiment_path.resolve()}")


if __name__ == "__main__":
    experimental_name = "exp_Z9"
    case_study = "9-bus"

    scenarios = ["scenario_A", "scenario_B"]

    alpha = {
        "scenario_A": 1.0,
        "scenario_B": 0.8
    }

    growth_rate = 1.00
    num_representative_days = 2

    power_flow_fidelity = {
        "scenario_A": "CP",
        "scenario_B": "DC"
    }

    relaxations = {
        "scenario_A": {"relax_second_stage": True, "unit_commitment": False},
        "scenario_B": {"relax_second_stage": False, "unit_commitment": True}
    }

    number_of_commitment = {
        "scenario_A": 12,
        "scenario_B": 12
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
        base_dir="."
    )