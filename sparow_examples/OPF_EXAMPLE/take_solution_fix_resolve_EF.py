import random
import math
import argparse
import munch
from forestlib.sp import stochastic_program
from forestlib.ef import ExtensiveFormSolver
from forestlib.ph import ProgressiveHedgingSolver
import json
from pyomo.environ import *
import sys
from IPython import embed
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import pi
import os
import pprint
from models_data import add_data_to_pyomo,model_data,model_data_3,model_data_ALL_HF,model_data_2,LF_builder,LF_builder_SHED,HF_builder
model_data_noisey = {
    "LF": {
        "scenarios": [
        {
            "ID": "Scen1_LF",
            "Probability": 1/8,
            "DEMAND":100,
            "data_dir":'../data/rts_noisey/scenarios/scenario_1.json',
        },
        {
            "ID": "Scen2_LF",
            "Probability": 1/8,
            "DEMAND":200,
            "data_dir":'../data/rts_noisey/scenarios/scenario_2.json',
        },
        {
            "ID": "Scen3_LF",
            "Probability": 1/8,
            "DEMAND":100,
            "data_dir":'../data/rts_noisey/scenarios/scenario_3.json',
        },
        {
            "ID": "Scen4_LF",
            "Probability": 1/8,
            "DEMAND":200,
            "data_dir":'../data/rts_noisey/scenarios/scenario_4.json',
        },
        {
            "ID": "Scen5_LF",
            "Probability": 1/8,
            "DEMAND":100,
            "data_dir":'../data/rts_noisey/scenarios/scenario_5.json',
        },
        {
            "ID": "Scen6_LF",
            "Probability": 1/8,
            "DEMAND":200,
            "data_dir":'../data/rts_noisey/scenarios/scenario_6.json',
        },
        {
            "ID": "Scen7_LF",
            "Probability": 1/8,
            "DEMAND":100,
            "data_dir":'../data/rts_noisey/scenarios/scenario_7.json',
        },
        {
            "ID": "Scen8_LF",
            "Probability": 1/8,
            "DEMAND":200,
            "data_dir":'../data/rts_noisey/scenarios/scenario_8.json',
        }
        ]
    },
}

print("-" * 60)
print("Running HF_EF")
print("-" * 60)
sp = stochastic_program(first_stage_variables=["ug[*,*]"])
#sp.initialize_application(app_data=app_data)
sp.initialize_model(
        name="HF", model_data=model_data_noisey["LF"], model_builder=HF_builder
    )


sp.initialize_bundles(scheme="single_bundle")
assert (len(sp.bundles) == 1), f"The extensive form should only have one bundle: {len(sp.bundles)}"

b = next(iter(sp.bundles))
M = sp.create_subproblem(b)

json_file_path= 'results_LF_EF_noisey.json'
json_file_path= 'results_LF_PH.json'
json_file_path= 'results_MF_SIM_PH.json'

with open(json_file_path, 'r') as file:
    data = json.load(file)

for i in M.s.index_set():
    #for j in range(0,len(data['null']['solutions']['0']['variables'])):
    for j in range(0,len(data['PH Iterations']['solutions']['0']['variables'])):
        #d_j=data['null']['solutions']['0']['variables'][j]
        d_j=data['PH Iterations']['solutions']['0']['variables'][j]
        name_with_index = d_j['name'][3:-1] 
        name, index = name_with_index.split(',')
        key = (name, int(index))
        if d_j['value']==0 or d_j['value'] == 1:
            M.s[i].ug[key].fix(round(d_j['value']))
        #M.s[i].ug[key].value =round (d_j['value']) 

#embed()
#M.obj.deactivate()
results = sp.solve(M, solver='gurobi',tee=True)


#solver = ExtensiveFormSolver()
#solver.set_options(solver="gurobi",loglevel="INFO")
#results = solver.solve(sp)
#results.write("results_HF_EF_sus.json", indent=4)
#print("Writing results to 'results.json'")
