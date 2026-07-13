import pprint  # for pretty-printing the results dict
#from sparow_examples.OPF_EXAMPLE.exp_1 import create_sp  # change this import to whatever exemplar you're running
from sparow.ph.ph_mpisppy import (
    ProgressiveHedgingSolver_MPISPPY,
)  

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
from egret.models.acopf import *
from egret.data.model_data import ModelData
from parameterized import parameterized
from egret.parsers.matpower_parser import create_ModelData
import time# import sparow wrapper around mpisppy

#sp = (
#    random_HF_LF1_grid_facilityloc()
#)  # replace sp obj with whatever exemplar you're running

random.seed(55)
dr=1e7
max_its=20
scenarios = ["1", "2", "3", "4", "5", "6", "7", "8"]
#scenarios = ["1", "2", "3", "4"]
probability = 1 / len(scenarios)
bnd=0.1
# Create the model_data dictionary
model_data_time = {
    'HF':{
    "scenarios": [
        {
            "ID": scenario,
            "Probability": probability,
            "DEMAND_1":1.0,
            "DEMAND_2":1.0+random.uniform(-bnd, bnd),
            "DEMAND_3":1.0+random.uniform(-bnd, bnd),
            "DEMAND_4":1.0+random.uniform(-bnd, bnd),
            "data_dir":'../../../../pglib-opf/pglib_opf_case118_ieee.m',
        }
        for scenario in scenarios
    ]
}
}


model_data_time_MF = {
    'HF': {
        "scenarios": model_data_time['HF']["scenarios"][:4]  # First 4 scenarios
    },
    'LF': {
        "scenarios": model_data_time['HF']["scenarios"][4:]  # Next 4 scenarios
    }
}



def LF_builder_multitime_period(data, args):
    test_case = data['data_dir']
    fm = ConcreteModel()

    fm.time_periods = Block([1, 2, 3, 4])
    objectives = []
    for time in fm.time_periods:

        fm.time_periods[time].m = ConcreteModel()

        model_data = create_ModelData(test_case)
        model_data.data['system']['load_mismatch_cost'] = 5000
        

        for load_name, load_info in model_data.data['elements']['load'].items():
            if load_info['in_service']:
                load_info['p_load'] *= data[f'DEMAND_{time}']


        fm.time_periods[time].m, md = create_copperplate_dispatch_approx_model(model_data, include_feasibility_slack=True)
        if hasattr(fm.time_periods[time].m, 'obj'):
            fm.time_periods[time].m.obj.deactivate()
            objectives.append(fm.time_periods[time].m.obj.expr)

    fm.obj = Objective(expr=sum(objectives), sense=minimize)

    return fm

def HF_builder_multitime_period(data, args):
    test_case = data['data_dir']
    fm = ConcreteModel()

    fm.time_periods = Block([1, 2, 3, 4])
    objectives = []
    for time in fm.time_periods:

        fm.time_periods[time].m = ConcreteModel()

        model_data = create_ModelData(test_case)
        model_data.data['system']['load_mismatch_cost'] = 5000
    
        for load_name, load_info in model_data.data['elements']['load'].items():
            if load_info['in_service']:
                load_info['p_load'] *= data[f'DEMAND_{time}']


        fm.time_periods[time].m, md = create_btheta_dcopf_model(model_data, include_feasibility_slack=True)
        if hasattr(fm.time_periods[time].m, 'obj'):
            fm.time_periods[time].m.obj.deactivate()
            objectives.append(fm.time_periods[time].m.obj.expr)

    fm.obj = Objective(expr=sum(objectives), sense=minimize)
    fm.ConstraintList = ConstraintList()
    for time in range(2, 5):  # Start from time 2 since we need the previous time step
            for g_name, g_info in model_data.data['elements']['generator'].items():
                pg_prev = fm.time_periods[time - 1].m.pg[g_name] 
                pg_curr = fm.time_periods[time].m.pg[g_name]  

                # Add constraint: pg can only go up or down by 10%
                fm.ConstraintList.add(expr=pg_curr <= pg_prev * 1.10)
                fm.ConstraintList.add(expr=pg_curr >= pg_prev * 0.90) 
    return fm
def uHF_builder_multitime_period(data, args):
    test_case = data['data_dir']
    fm = ConcreteModel()

    fm.time_periods = Block([1, 2, 3, 4])
    objectives = []
    for time in fm.time_periods:

        fm.time_periods[time].m = ConcreteModel()

        model_data = create_ModelData(test_case)
        model_data.data['system']['load_mismatch_cost'] = 500
        model_data.data['system']['q_load_mismatch_cost'] = 0
    
        for load_name, load_info in model_data.data['elements']['load'].items():
            if load_info['in_service']:
                load_info['p_load'] *= data[f'DEMAND_{time}']


        fm.time_periods[time].m, md = create_rsv_acopf_model(model_data, include_feasibility_slack=True)
        if hasattr(fm.time_periods[time].m, 'obj'):
            fm.time_periods[time].m.obj.deactivate()
            objectives.append(fm.time_periods[time].m.obj.expr)

    fm.obj = Objective(expr=sum(objectives), sense=minimize)
    fm.ConstraintList = ConstraintList()
    for time in range(2, 5):  # Start from time 2 since we need the previous time step
            for g_name, g_info in model_data.data['elements']['generator'].items():
                pg_prev = fm.time_periods[time - 1].m.pg[g_name] 
                pg_curr = fm.time_periods[time].m.pg[g_name]  

                # Add constraint: pg can only go up or down by 10%
                fm.ConstraintList.add(expr=pg_curr <= pg_prev * 1.10)
                fm.ConstraintList.add(expr=pg_curr >= pg_prev * 0.90) 
    return fm

sp = stochastic_program(first_stage_variables=["time_periods[1].m.pg[*]"])
sp.initialize_model(
    name="HF", model_data=model_data_time["HF"], model_builder=HF_builder_multitime_period
)

solver = (
    ProgressiveHedgingSolver_MPISPPY()
)  # solving with the sparow wrapper around mpisppy
solver.set_options(
    solver="ipopt",  # not sure we support other solvers right now
    max_iterations=100,  # i think this is 100 by default?
    loglevel="INFO",  # can replace with DEBUG, VERBOSE, etc.
    default_rho=dr,  # rho by default will already be 1.5
    mpisppy_options=[
        "--lagrangian",
        "--xhatshuffle",
        "--rel-gap=0.01",
        "--default-rho=1e7",
    ],  # can customize these also
)

results_mpi = solver.solve(sp, solver="ipopt")  # solution obj
if getattr(solver, "mpi_rank", 0) == 0:  # all the information gets sent to rank 0
    pprint.pprint(results_mpi.to_dict())  # pretty-print results