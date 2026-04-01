import pytest

from sparow.ef import ExtensiveFormSolver
from sparow.ph import ProgressiveHedgingSolver
from IPython import embed
import pyomo.opt
from pyomo.common import unittest
from IPython import embed
from sparow.sp.util import relax_second_stage

solvers = set(pyomo.opt.check_available_solvers("gurobi"))

try:
    from sparow_examples.gtep_9bus.load_scenarios_w_Power_fidelity import create_sp

    dummy_available = True
except:
    dummy_available = False
rd = {'low_alpha':True,'high_alpha':False}

sp = create_sp()
sp.add_transformation(relax_second_stage,relax_dict=rd)

solver = ProgressiveHedgingSolver()
solver.set_options(solver="gurobi", loglevel="INFO")
#solver = ProgressiveHedgingSolver()
#solver.set_options(solver="gurobi", loglevel="INFO",max_iterations=30)
#sp.create_subproblem(b='model_low_alpha',cached=True)
#munch_object = solver.solve_and_return_EF(sp)

results = solver.solve(sp)
results_dict = results.to_dict()
#embed()
soln = next(iter(results_dict["solutions"].values()))

obj_val = soln["objectives"][0]["value"]
print(obj_val)
print(dummy_available)
#embed()

assert obj_val == pytest.approx(77450369.99037874, 0.01)
