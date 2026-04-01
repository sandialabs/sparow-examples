import pytest
import time
from sparow.ef import ExtensiveFormSolver
from sparow.ph import ProgressiveHedgingSolver
from IPython import embed
import pyomo.opt
from pyomo.common import unittest
from IPython import embed
from sparow.sp.util import relax_second_stage

solvers = set(pyomo.opt.check_available_solvers("baron"))

try:
    from sparow_examples.gtep_9bus.load_scenarios_w_relaxed_second_stage import create_sp

    dummy_available = True
except:
    dummy_available = False
rd = {'low_alpha':True,'high_alpha':True}

# RUN Progressive Hedging Standard

sp = create_sp()
sp.add_transformation(relax_second_stage,relax_dict=rd)
#sp.transform_subproblem=relax_second_stage
solver = ProgressiveHedgingSolver()
solver.set_options(solver="scip", loglevel="INFO",max_iterations=10)
PH_time_start=time.time()
results = solver.solve(sp)
results_dict = results.to_dict()
soln = next(iter(results_dict["solutions"].values()))
PH_time_end=time.time()
print("PH time:")
print(PH_time_end-PH_time_start)
obj_val = soln["objectives"][0]["value"]
print(obj_val)
print(dummy_available)


# RUN Extensive Form Standard

sp = create_sp()
sp.add_transformation(relax_second_stage,relax_dict=rd)
solver = ExtensiveFormSolver()
solver.set_options(solver="gurobi", loglevel="INFO")
EF_time_start=time.time()
#solver = ProgressiveHedgingSolver()
#solver.set_options(solver="gurobi", loglevel="INFO",max_iterations=30)
#sp.create_subproblem(b='model_low_alpha',cached=True)
#munch_object = solver.solve_and_return_EF(sp)

results = solver.solve(sp)
results_dict = results.to_dict()
#embed()
soln = next(iter(results_dict["solutions"].values()))

EF_time_end=time.time()
print("EF time:")
print(EF_time_end-EF_time_start)

obj_val = soln["objectives"][0]["value"]
print(obj_val)
print(dummy_available)
#embed()

assert obj_val == pytest.approx(77450369.99037874, 0.01)
