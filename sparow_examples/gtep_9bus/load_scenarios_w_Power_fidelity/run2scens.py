import pytest

from sparow.ef import ExtensiveFormSolver
from sparow.ph import ProgressiveHedgingSolver
from IPython import embed
import pyomo.opt
from pyomo.common import unittest

solvers = set(pyomo.opt.check_available_solvers("gurobi"))

try:
    from sparow_examples.gtep_9bus.load_scenarios import create_sp

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
print(dummy_available)

assert obj_val == pytest.approx(285632.11, 0.01)
