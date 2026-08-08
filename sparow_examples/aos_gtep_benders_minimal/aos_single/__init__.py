
# sparow_examples.aos_gtep_9bus.load_scenarios

from sparow.sp import stochastic_program
import importlib


DEFAULT_APP_DATA = {
    "stages": 3,
    "num_reps": 2,
    "len_reps": 24,
    "num_commit": 24,
    "num_dispatch": 1,
}
model_data = {"scenarios": [{"ID": "single", "Demand": 1.0, "Probability": 1.0,"alpha":1.0}]}
#model_data = {"scenarios": [{"ID": "low_alpha", "Demand": 1.0, "Probability": 0.5,"alpha":1.0},{"ID": "high_alpha", "Demand": 1.0, "Probability": 0.5,"alpha":0.90}]}


def model_builder(data, args):
    num_stages = data["stages"]
    num_rep_days = data["num_reps"]
    len_rep_days = data["len_reps"]
    num_commit_p = data["num_commit"]
    num_disp = data["num_dispatch"]
    alpha = data["alpha"]
                              
    scenario = importlib.import_module("sparow_examples.aos_gtep_9bus.aos_single."+data['ID'])
    return scenario.create_gtep_model(
        num_stages=num_stages,
        num_rep_days=num_rep_days,
        len_rep_days=len_rep_days,
        num_commit_p=num_commit_p,
        num_disp=num_disp,
        alpha= alpha
    )


def create_sp(app_data=None, **size_overrides):
    
    updated_app_data = dict(DEFAULT_APP_DATA if app_data is None else app_data)
    updated_app_data.update(size_overrides)
    
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
    sp.initialize_application(app_data=updated_app_data)
    sp.initialize_model(
        #name="model",
        model_data=model_data, model_builder=model_builder
    )
    return sp
