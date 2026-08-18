import json
import os


# ---------------------------------------------------------------------
# Base scenario definitions
# ---------------------------------------------------------------------

BASE_SCENARIOS_12 = [
    ("Jan27", "../data/rts_gmlc/2020-01-27.json", 100),
    ("Feb09", "../data/rts_gmlc/2020-02-09.json", 200),
    ("Mar05", "../data/rts_gmlc/2020-03-05.json", 200),
    ("Apr03", "../data/rts_gmlc/2020-04-03.json", 200),
    ("May05", "../data/rts_gmlc/2020-05-05.json", 200),
    ("June09", "../data/rts_gmlc/2020-06-09.json", 200),
    ("July06", "../data/rts_gmlc/2020-07-06.json", 200),
    ("Aug12", "../data/rts_gmlc/2020-08-12.json", 200),
    ("Sep20", "../data/rts_gmlc/2020-09-20.json", 200),
    ("Oct27", "../data/rts_gmlc/2020-10-27.json", 200),
    ("Nov25", "../data/rts_gmlc/2020-11-25.json", 200),
    ("Dec23", "../data/rts_gmlc/2020-12-23.json", 200),
]

BASE_SCENARIOS_3 = [
    ("Jan27", "../data/rts_gmlc/2020-01-27.json", 100),
    ("Feb09", "../data/rts_gmlc/2020-02-09.json", 200),
    ("Mar05", "../data/rts_gmlc/2020-03-05.json", 200),
]

BASE_SCENARIOS_2_LF = [
    ("Jan27", "../data/rts_gmlc/2020-02-09.json", 100),
    ("Feb09", "../data/rts_gmlc/2020-03-05.json", 200),
]

BASE_SCENARIOS_2_HF = [
    ("Jan27", "../data/rts_gmlc/2020-06-09.json", 100),
    ("Feb09", "../data/rts_gmlc/2020-07-06.json", 200),
]

BASE_SCENARIOS_ALL_HF = [
    ("Feb09", "../data/rts_gmlc/2020-02-09.json", 100),
    ("Mar05", "../data/rts_gmlc/2020-03-05.json", 200),
    ("Jun09", "../data/rts_gmlc/2020-06-09.json", 100),
    ("Jul06", "../data/rts_gmlc/2020-07-06.json", 200),
]


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _build_fidelity_block(base_scenarios, fidelity_name):
    probability = 1 / len(base_scenarios)
    return {
        fidelity_name: {
            "scenarios": [
                {
                    "ID": f"{label}_{fidelity_name}",
                    "Probability": probability,
                    "DEMAND": demand,
                    "data_dir": data_dir,
                }
                for label, data_dir, demand in base_scenarios
            ]
        }
    }


def _build_lf_hf_blocks(base_scenarios_lf, base_scenarios_hf=None):
    if base_scenarios_hf is None:
        base_scenarios_hf = base_scenarios_lf

    prob_lf = 1 / len(base_scenarios_lf)
    prob_hf = 1 / len(base_scenarios_hf)

    return {
        "LF": {
            "scenarios": [
                {
                    "ID": f"{label}_LF",
                    "Probability": prob_lf,
                    "DEMAND": demand,
                    "data_dir": data_dir,
                }
                for label, data_dir, demand in base_scenarios_lf
            ]
        },
        "HF": {
            "scenarios": [
                {
                    "ID": f"{label}_HF",
                    "Probability": prob_hf,
                    "DEMAND": demand,
                    "data_dir": data_dir,
                }
                for label, data_dir, demand in base_scenarios_hf
            ]
        },
    }


def load_demand_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["demand"]


def attach_demand_profiles(model_data, scenario_key="HF", demand_field="Demand", warn_missing=True):
    """
    Adds scenario[demand_field] = <demand time series from JSON file>
    for every scenario in model_data[scenario_key]["scenarios"].
    """
    for scenario in model_data[scenario_key]["scenarios"]:
        data_dir = scenario["data_dir"]
        if os.path.exists(data_dir):
            scenario[demand_field] = load_demand_data(data_dir)
        elif warn_missing:
            print(f"Warning: {data_dir} does not exist.")

    return model_data


def attach_demand_profiles_all(model_data, demand_field="Demand", warn_missing=True):
    """
    Adds demand profiles to every fidelity block present in the model_data dict.
    """
    for fidelity_name in model_data:
        attach_demand_profiles(
            model_data=model_data,
            scenario_key=fidelity_name,
            demand_field=demand_field,
            warn_missing=warn_missing,
        )
    return model_data


# ---------------------------------------------------------------------
# Public scenario factory functions
# ---------------------------------------------------------------------

def get_model_data():
    """
    Equivalent of the old model_data variable.
    """
    return _build_lf_hf_blocks(BASE_SCENARIOS_12)


def get_model_data_3():
    """
    Equivalent of the old model_data_3 variable.
    """
    return _build_lf_hf_blocks(BASE_SCENARIOS_3)


def get_model_data_2():
    """
    Equivalent of the old model_data_2 variable.
    """
    return _build_lf_hf_blocks(BASE_SCENARIOS_2_LF, BASE_SCENARIOS_2_HF)


def get_model_data_all_hf():
    """
    Equivalent of the old model_data_ALL_HF variable.
    """
    return _build_fidelity_block(BASE_SCENARIOS_ALL_HF, "HF")


def build_noisy_hf_scenarios(
    dir_start="../../data/rts_noisey/scenarios_2/",
    n_scenarios=8,
    add_demand_profiles=True,
    demand_field="Demand",
):
    """
    Builds the noisy HF scenario block used in scripts like HF_EF.py.

    Example scenario IDs:
        Scen1_HF, Scen2_HF, ..., Scen8_HF

    Demand values follow your current alternating pattern:
        odd scenarios -> 100
        even scenarios -> 200
    """
    probability = 1 / n_scenarios

    model_data_noisey = {
        "HF": {
            "scenarios": [
                {
                    "ID": f"Scen{i}_HF",
                    "Probability": probability,
                    "DEMAND": 100 if i % 2 == 1 else 200,
                    "data_dir": os.path.join(dir_start, f"scenario_{i}.json"),
                }
                for i in range(1, n_scenarios + 1)
            ]
        }
    }

    if add_demand_profiles:
        attach_demand_profiles(
            model_data_noisey,
            scenario_key="HF",
            demand_field=demand_field,
        )

    return model_data_noisey