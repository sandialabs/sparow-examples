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
import time

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
model_data_ALL_HF_time = {
    "HF": {
        "scenarios": [
        {
            "ID": "Feb09_HF",
            "Probability": 1/4,
            "DEMAND_1":1.0,
            "DEMAND_2":1.1,
            "DEMAND_3":1.2,
            "DEMAND_4":1.3,
            "data_dir":'../../../../pglib-opf/pglib_opf_case118_ieee.m',
        },
        {
            "ID": "Mar05_HF",
            "Probability": 1/4,
            "DEMAND_1":1.0,
            "DEMAND_2":0.9,
            "DEMAND_3":0.8,
            "DEMAND_4":0.7,
            "data_dir":'../../../../pglib-opf/pglib_opf_case118_ieee.m',
        },
        {
            "ID": "Jun09_HF",
            "Probability": 1/4,
            "DEMAND_1":1.0,
            "DEMAND_2":1.1,
            "DEMAND_3":1.0,
            "DEMAND_4":0.9,
            "data_dir":'../../../../pglib-opf/pglib_opf_case118_ieee.m',
        },
        {
            "ID": "Jul06_HF",
            "Probability": 1/4,
            "DEMAND_1":1.0,
            "DEMAND_2":0.9,
            "DEMAND_3":1.0,
            "DEMAND_4":1.1,
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


        fm.time_periods[time].m, md = create_psv_acopf_model(model_data, include_feasibility_slack=True)
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


# Initialize a dictionary to store results
results_summary = {
    "HF_EF": {"cpu_time": None, "objective": None},
    "LF_EF": {"cpu_time": None, "objective": None},
    "MF_EF": {"cpu_time": None, "objective": None},
}

# ----------------------- LF EF ----------------------- #
print("-" * 60)
print("Running DCOPF LF_EF")
print("-" * 60)
sp = stochastic_program(first_stage_variables=["time_periods[1].m.pg[*]"])
sp.initialize_model(
    name="HF", model_data=model_data_time["HF"], model_builder=HF_builder_multitime_period
)

solver = ExtensiveFormSolver()
solver.set_options(solver="ipopt", loglevel="INFO")
start = time.time()
results = solver.solve(sp)
end = time.time()

# Save CPU time and objective for LF_EF
results_summary["LF_EF"]["cpu_time"] = end - start
results_summary["LF_EF"]["objective"] = results.to_dict()['solutions'][0]['objectives'][0]['value']
results_dict_LF=results.to_dict()

print('TIME CPU LF_EF:', results_summary["LF_EF"]["cpu_time"])
print("Writing results to 'results.json'")

# ----------------------- HF EF ----------------------- #
print("-" * 60)
print("Running ACOPF HF_EF")
print("-" * 60)
sp = stochastic_program(first_stage_variables=["time_periods[1].m.pg[*]"])
sp.initialize_model(
    name="HF", model_data=model_data_time["HF"], model_builder=uHF_builder_multitime_period
)

solver = ExtensiveFormSolver()
solver.set_options(solver="ipopt", loglevel="INFO")
start = time.time()
results = solver.solve(sp)
end = time.time()

# Save CPU time and objective for HF_EF
results_summary["HF_EF"]["cpu_time"] = end - start
results_summary["HF_EF"]["objective"] = results.to_dict()['solutions'][0]['objectives'][0]['value']
results_dict_HF=results.to_dict()

print('TIME CPU HF_EF:', results_summary["HF_EF"]["cpu_time"])
print("Writing results to 'results.json'")

# ----------------------- MF EF ----------------------- #
print("-" * 60)
print("Running ACOPF/DCOPF MF_EF")
print("-" * 60)
sp = stochastic_program(first_stage_variables=["time_periods[1].m.pg[*]"])
sp.initialize_model(
    name="HF", model_data=model_data_time_MF["HF"], model_builder=uHF_builder_multitime_period
)
sp.initialize_model(
    name="LF", model_data=model_data_time_MF["LF"], model_builder=HF_builder_multitime_period
)


sp.initialize_bundles(
        scheme="mf_random",
        LF=2,
        seed=1234567890,
    )

solver = ExtensiveFormSolver()
solver.set_options(solver="ipopt", loglevel="INFO")
start = time.time()
results = solver.solve(sp)
end = time.time()

# Save CPU time and objective for MF_EF
results_summary["MF_EF"]["cpu_time"] = end - start
results_summary["MF_EF"]["objective"] = results.to_dict()['solutions'][0]['objectives'][0]['value']
results_dict_MF=results.to_dict()
print('TIME CPU MF_EF:', results_summary["MF_EF"]["cpu_time"])
print("Writing results to 'results.json'")

# Print all results at the end
print("\nSummary of Results:")
print("-" * 60)
for method, result in results_summary.items():
    print(f"{method}:")
    print(f"  CPU Time: {result['cpu_time']:.4f} seconds")
    print(f"  Objective Value: {result['objective']:.4f}")
print("-" * 60)



hf_variables = {var['name']: var['value'] for var in results_dict_HF['solutions'][0]['variables']}

# Function to compute percent error
def compute_percent_error(true_values, comparison_values):
    percent_errors = {}
    for name, true_value in true_values.items():
        if name in comparison_values:
            comparison_value = comparison_values[name]
            if true_value != 0:  # Avoid division by zero
                percent_error = abs(true_value - comparison_value) / abs(true_value) * 100
            else:
                percent_error = float('inf')  # Handle case where true value is zero
            percent_errors[name] = percent_error
    return percent_errors

# Extract variable values from LF_EF results
lf_variables = {var['name']: var['value'] for var in results_dict_LF['solutions'][0]['variables']}
lf_percent_errors = compute_percent_error(hf_variables, lf_variables)

# Extract variable values from MF_EF results
mf_variables = {var['name']: var['value'] for var in results_dict_MF['solutions'][0]['variables']}
mf_percent_errors = compute_percent_error(hf_variables, mf_variables)

# Print the percent errors
print("\nPercent Errors for LF_EF compared to HF_EF:")
#for name, error in lf_percent_errors.items():
#    print(f"{name}: {error:.2f}%")

print("\nPercent Errors for MF_EF compared to HF_EF:")
#for name, error in mf_percent_errors.items():
#    print(f"{name}: {error:.2f}%")


# Resolve HF_EF with LF_EF Solution

print("-" * 60)
print("Running ACOPF HF_EF")
print("-" * 60)
sp = stochastic_program(first_stage_variables=["time_periods[1].m.pg[*]"])
sp.initialize_model(
    name="HF", model_data=model_data_time["HF"], model_builder=uHF_builder_multitime_period
)

sp.initialize_bundles(scheme="single_bundle")
assert (len(sp.bundles) == 1), f"The extensive form should only have one bundle: {len(sp.bundles)}"

b = next(iter(sp.bundles))
M = sp.create_subproblem(b)

for i in M.s.index_set():
    for j in range(0,len(results_dict_LF['solutions'][0]['variables'])):

        d_j=results_dict_LF['solutions'][0]['variables'][j]
        name_with_index = d_j['name']

        M.s[i].time_periods[1].m.pg[str(j+1)].fix((d_j['value']))



results = sp.solve(M, solver='ipopt',tee=True)

# Resolve HF_EF with MF_EF Solution

print("-" * 60)
print("Running ACOPF HF_EF")
print("-" * 60)
sp = stochastic_program(first_stage_variables=["time_periods[1].m.pg[*]"])
sp.initialize_model(
    name="HF", model_data=model_data_time["HF"], model_builder=uHF_builder_multitime_period
)

sp.initialize_bundles(scheme="single_bundle")
assert (len(sp.bundles) == 1), f"The extensive form should only have one bundle: {len(sp.bundles)}"

b = next(iter(sp.bundles))
M = sp.create_subproblem(b)

for i in M.s.index_set():
    for j in range(0,len(results_dict_MF['solutions'][0]['variables'])):

        d_j=results_dict_MF['solutions'][0]['variables'][j]
        name_with_index = d_j['name']

        M.s[i].time_periods[1].m.pg[str(j+1)].fix((d_j['value']))



results = sp.solve(M, solver='ipopt',tee=True)