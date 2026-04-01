#
# Setup BUS-9_ALPHA-1~1.2_GROWTH-1.0_REP-4_FIDELITY-CP~DC_RELAX-None_COMMITMENT-4
#
# 
# 

import os
import shutil
import string

# The example name
name = "A9"
scenarios = ["low_alpha", "high_alpha"]

if not os.path.exists(name):
    os.mkdir(name)

for scen in scenarios:
    dirname = os.path.join(name, scen)
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    shutil.copytree("model", dirname)


module_root = string.Template(
    """
# sparow_examples.organized_gtep_runs.A9

from sparow.sp import stochastic_program
import importlib


app_data = {
    "stages": 3,
    "num_reps": 4,
    "len_reps": 1,
    "num_commit": 4,
    "num_dispatch": 1,
}
model_data = {"scenarios": [{"ID": "low_alpha", "Demand": 1.0, "Probability": 0.5,"alpha":1.0,"PF":"CP"},{"ID": "high_alpha", "Demand": 1.0, "Probability": 0.5,"alpha":1.2,"PF":"DC"}]}


def model_builder(data, args):
    num_stages = data["stages"]
    num_rep_days = data["num_reps"]
    len_rep_days = data["len_reps"]
    num_commit_p = data["num_commit"]
    num_disp = data["num_dispatch"]
    alpha = data["alpha"]
    PF = data["PF"]
                              
    scenario = importlib.import_module("sparow_examples.organized_gtep_runs.$name."+data['ID'])
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
"""
).substitute(name=name)

with open(os.path.join(name, "__init__.py"), "w") as OUTPUT:
    OUTPUT.write(module_root)
