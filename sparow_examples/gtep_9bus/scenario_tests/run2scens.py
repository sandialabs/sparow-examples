import pytest

from sparow.ef import ExtensiveFormSolver
from sparow.ph import ProgressiveHedgingSolver
from IPython import embed
import pyomo.opt
from pyomo.common import unittest
from IPython import embed
from sparow.sp.util import relax_second_stage
import time

solvers = set(pyomo.opt.check_available_solvers("gurobi"))

try:
    from sparow_examples.gtep_9bus.scenario_tests import create_sp

    dummy_available = True
except:
    dummy_available = False
#rd = {"1":False, "2":False, "3":False, "4":False, "5":False, "6":False, "7":False, "8":False}
rd = {"1":True, "2":True, "3":True, "4":True, "5":True, "6":True, "7":True, "8":True}

sp = create_sp()
sp.add_transformation(relax_second_stage,relax_dict=rd)

#solver = ProgressiveHedgingSolver()
solver = ExtensiveFormSolver()
solver.set_options(solver="gurobi", loglevel="INFO")
#solver = ProgressiveHedgingSolver()
#solver.set_options(solver="gurobi", loglevel="INFO",max_iterations=30)
#sp.create_subproblem(b='model_low_alpha',cached=True)
#munch_object = solver.solve_and_return_EF(sp)
start= time.time()
results = solver.solve(sp)
end= time.time()
print('TIME CPU')
print(end-start)
results_dict = results.to_dict()
#embed()
soln = next(iter(results_dict["solutions"].values()))

obj_val = soln["objectives"][0]["value"]
print(obj_val)
print(dummy_available)
#embed()

assert obj_val == pytest.approx(77450369.99037874, 0.01)
