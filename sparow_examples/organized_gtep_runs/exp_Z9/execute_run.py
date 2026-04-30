import pytest
from sparow.ef import ExtensiveFormSolver
from sparow.ph import ProgressiveHedgingSolver
import pyomo.opt
from pyomo.common import unittest
from sparow.sp.util import relax_second_stage

solvers = set(pyomo.opt.check_available_solvers("gurobi"))

try:
    from sparow_examples.organized_gtep_runs.exp_Z9 import create_sp

    dummy_available = True
except:
    dummy_available = False

relaxations = {'scenario_A': {'relax_second_stage': True, 'unit_commitment': False}, 'scenario_B': {'relax_second_stage': False, 'unit_commitment': True}}
rd = {k: v["relax_second_stage"] for k, v in relaxations.items()}
sp = create_sp()
sp.add_transformation(relax_second_stage, relax_dict=rd)
solver = ExtensiveFormSolver()
solver.set_options(solver="gurobi", loglevel="INFO")
results = solver.solve(sp)
results_dict = results.to_dict()

soln = next(iter(results_dict["solutions"].values()))

obj_val = soln["objectives"][0]["value"]
print(obj_val)
