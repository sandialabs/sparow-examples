
# sparow_examples.gtep_9bus.scenario_tests

from sparow.sp import stochastic_program
import importlib
import random

scenarios = ["1", "2", "3", "4", "5", "6", "7", "8"]
probability = 1 / len(scenarios)

app_data = {
    "stages": 3,
    "num_reps": 4,
    "len_reps": 24,
    "num_commit": 3,
    "num_dispatch": 1,
}

model_data = {
    "scenarios": [
        {
            "ID": scenario,
            "Demand": random.uniform(0.8, 1.2),  # Random demand between 0.8 and 1.2
            "Probability": probability,
            "alpha": random.uniform(0.9, 4.0),  # Random alpha between 0.9 and 1.0
            "PF":"DC"
        }
        for scenario in scenarios
    ]
}

def model_builder(data, args):
    num_stages = data["stages"]
    num_rep_days = data["num_reps"]
    len_rep_days = data["len_reps"]
    num_commit_p = data["num_commit"]
    num_disp = data["num_dispatch"]
    alpha = data["alpha"]
    PF = data["PF"]
                              
    scenario = importlib.import_module("sparow_examples.gtep_9bus.scenario_tests."+data['ID'])
    return scenario.create_gtep_model(
        num_stages=num_stages,
        num_rep_days=num_rep_days,
        len_rep_days=len_rep_days,
        num_commit_p=num_commit_p,
        num_disp=num_disp,
        alpha= alpha,
        flow_model = PF
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
