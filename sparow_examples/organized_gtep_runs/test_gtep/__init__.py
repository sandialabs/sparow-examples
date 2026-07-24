# sparow_examples.organized_gtep_runs.test_gtep

from sparow.sp import stochastic_program
import importlib


app_data = {'stages': 3, 'len_reps': 24, 'num_dispatch': 1}
model_data = {'scenarios': [{'ID': 'scenario_A', 'Demand': 1.0, 'Probability': 1.0, 'alpha': 1.0, 'PF': 'DC', 'num_reps': 4, 'representative_dates': ['2020-01-28 00:00', '2020-04-23 00:00', '2020-07-05 00:00', '2020-10-14 00:00'], 'representative_weights': [1, 1, 1, 1], 'num_commit': 24, 'include_commitment': True}]}


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
        "sparow_examples.organized_gtep_runs.test_gtep." + data["ID"]
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
            "investmentStage[*].storOperational[*].binary_indicator_var",
            "investmentStage[*].storInstalled[*].binary_indicator_var",
            "investmentStage[*].storRetired[*].binary_indicator_var",
            "investmentStage[*].storDisabled[*].binary_indicator_var",
            "investmentStage[*].storExtended[*].binary_indicator_var",
        ]
    )
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="model", model_data=model_data, model_builder=model_builder
    )
    return sp
