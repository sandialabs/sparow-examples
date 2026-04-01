import random
import math
import argparse
import munch
from sparow.sp import stochastic_program
from sparow.ef import ExtensiveFormSolver
from sparow.ph import ProgressiveHedgingSolver
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
from egret.models.copperplate_dispatch import *
from egret.models.dcopf import *
from egret.data.model_data import ModelData
from parameterized import parameterized
from egret.parsers.matpower_parser import create_ModelData

model_data_ALL_HF = {
    "HF": {
        "scenarios": [
        {
            "ID": "Feb09_HF",
            "Probability": 1/4,
            "DEMAND":1.0,
            "data_dir":'../../../../pglib-opf/pglib_opf_case118_ieee.m',
        },
        {
            "ID": "Mar05_HF",
            "Probability": 1/4,
            "DEMAND":1.0,
            "data_dir":'../../../../pglib-opf/pglib_opf_case118_ieee.m',
        },
        {
            "ID": "Jun09_HF",
            "Probability": 1/4,
            "DEMAND":1.0,
            "data_dir":'../../../../pglib-opf/pglib_opf_case118_ieee.m',
        },
        {
            "ID": "Jul06_HF",
            "Probability": 1/4,
            "DEMAND":1.0,
            "data_dir":'../../../../pglib-opf/pglib_opf_case118_ieee.m',
        }
        ]
    },
}
def LF_builder(data,args):
    test_case = data['data_dir']
    model_data = create_ModelData(test_case)
    for load_name, load_info in model_data.data['elements']['load'].items():
        if load_info['in_service']: 
            load_info['p_load'] *= data['DEMAND']
    m,md=create_copperplate_dispatch_approx_model(model_data)
    return m
def HF_builder(data,args):
    test_case = data['data_dir']
    model_data = create_ModelData(test_case)
    model_data.data['system']['load_mismatch_cost']=5000
    for load_name, load_info in model_data.data['elements']['load'].items():
        if load_info['in_service']: 
            load_info['p_load'] *= data['DEMAND']
    m,md=create_btheta_dcopf_model(model_data,include_feasibility_slack=True)
    return m

#test_case = '../../../pglib-opf/pglib_opf_case118_ieee.m'
#model_data = create_ModelData(test_case)
#m,md=create_btheta_dcopf_model(model_data)

print("-" * 60)
print("Running HF_EF")
print("-" * 60)
sp = stochastic_program(first_stage_variables=["pg[*]"])
#sp.initialize_application(app_data=app_data)
sp.initialize_model(
    name="HF", model_data=model_data_ALL_HF["HF"], model_builder=HF_builder
)
#solver = ProgressiveHedgingSolver()
solver = ExtensiveFormSolver()
solver.set_options(solver="gurobi",loglevel="INFO")
results = solver.solve(sp)
#results.write("results_HF_EF_sus.json", indent=4)
print("Writing results to 'results.json'")
results_dict = results.to_dict()
print(results_dict)
