from egret.models.copperplate_dispatch import *
from egret.models.dcopf import *
from egret.models.acopf import *
from egret.data.model_data import ModelData
from egret.parsers.matpower_parser import create_ModelData
from pyomo.environ import *

data_dir='../../../../pglib-opf/pglib_opf_case14_ieee.m' #Replace with your actual PGLIB Data file

test_case = data_dir
model_data = create_ModelData(test_case)

m,md=create_copperplate_dispatch_approx_model(model_data)

SolverFactory("gurobi").solve(m)
m.pprint()

model_data = create_ModelData(test_case)

m,md=create_btheta_dcopf_model(model_data)

SolverFactory("gurobi").solve(m)
m.pprint()

model_data = create_ModelData(test_case)

m,md=create_rsv_acopf_model(model_data)

SolverFactory("ipopt").solve(m)
m.pprint()

