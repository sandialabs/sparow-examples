import pytest
from sparow.ef import ExtensiveFormSolver
from sparow.ph import ProgressiveHedgingSolver
import pyomo.opt
from pyomo.common import unittest
from sparow.sp.util import relax_second_stage
import time
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from post_process_gtep_solution import post_process

solvers = set(pyomo.opt.check_available_solvers("gurobi"))

try:
    from sparow_examples.organized_gtep_runs.exp_SFSS_B9 import create_sp

    dummy_available = True
except:
    dummy_available = False

relaxations = {'scenario_A': {'relax_second_stage': False, 'unit_commitment': True}}
rd = {k: v["relax_second_stage"] for k, v in relaxations.items()}

script_start = time.perf_counter()
sp = create_sp()
sp.add_transformation(relax_second_stage, relax_dict=rd)
solver = ExtensiveFormSolver()
solver.set_options(solver="gurobi", loglevel="INFO")

solve_start = time.perf_counter()
res_munch = solver.solve_and_return_EF(sp)
results = res_munch.solutions
solve_end = time.perf_counter()

results_dict = results.to_dict()

soln = next(iter(results_dict["solutions"].values()))

obj_val = soln["objectives"][0]["value"]
total_end = time.perf_counter()
print(f"Objective value: {obj_val}")
print(f"Solve time (seconds): {solve_end - solve_start:.4f}")
print(f"Total runtime (seconds): {total_end - script_start:.4f}")

mod_object=res_munch.model.s['model','scenario_A']
post_process(mod_object)
