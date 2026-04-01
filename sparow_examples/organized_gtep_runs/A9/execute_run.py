import pytest

from sparow.ef import ExtensiveFormSolver
from sparow.ph import ProgressiveHedgingSolver
from IPython import embed
import pyomo.opt
from pyomo.common import unittest

solvers = set(pyomo.opt.check_available_solvers("gurobi"))

try:
    from sparow_examples.organized_gtep_runs.A9 import create_sp

    dummy_available = True
except:
    dummy_available = False

sp = create_sp()
solver = ExtensiveFormSolver()
solver.set_options(solver="gurobi", loglevel="INFO")
results = solver.solve(sp)
results_dict = results.to_dict()

soln = next(iter(results_dict["solutions"].values()))

obj_val = soln["objectives"][0]["value"]
print(obj_val)


