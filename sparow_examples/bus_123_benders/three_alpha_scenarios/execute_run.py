import pytest
from sparow.ef import ExtensiveFormSolver
from sparow.ph import ProgressiveHedgingSolver
from sparow.benders import BendersSolver
import pyomo.opt
from pyomo.common import unittest
from sparow.sp.util import relax_second_stage
import time
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#from post_process_gtep_solution import post_process

solvers = set(pyomo.opt.check_available_solvers("gurobi"))

try:
    from sparow_examples.bus_123_benders.three_alpha_scenarios import create_sp

    dummy_available = True
except:
    dummy_available = False


relaxations = {'scenario_A': {'relax_second_stage': True, 'unit_commitment': True}, 'scenario_B': {'relax_second_stage': True, 'unit_commitment': True}, 'scenario_C': {'relax_second_stage': True, 'unit_commitment': True}}
rd = {k: v["relax_second_stage"] for k, v in relaxations.items()}

script_start = time.perf_counter()
sp = create_sp()
sp.add_transformation(relax_second_stage, relax_dict=rd)
solver = BendersSolver()

TIME_LIMIT = 1200.0
CONVERGENCE_TOL = 1e-3
MAX_ITERATIONS = 100
ETA_LOWER_BOUND_DEFAULT = -1e6

eta_bounds_map = {b: (ETA_LOWER_BOUND_DEFAULT, None) for b in sp.bundles}


def _on_iteration(data):
    n_cuts = len(data.cuts_added) if data.cuts_added is not None else 0
    print(
        f"  --- Benders iteration {data.iter_idx}  cuts_added={n_cuts} ---",
        flush=True,
    )

solver.set_options(
    solver="gurobi_persistent",
    subproblem_solver="gurobi_persistent",
    max_iterations=MAX_ITERATIONS,
    is_persistent_solver=True,
    allow_infeasible_subproblems=True,
    loglevel="INFO",
    solver_options={"TimeLimit": TIME_LIMIT, "timelimit": TIME_LIMIT},
)

print(
    f"--- Calling solve_and_return_model "
    f"(TimeLimit={TIME_LIMIT}s, convergence_tol={CONVERGENCE_TOL}) ---",
    flush=True,
)

solve_start = time.perf_counter()
res_munch = solver.solve_and_return_model(
    sp,
    eta_bounds_map,
    #subproblem_transforms=[relax_second_stage],
    master_transforms=None,
    on_iteration=_on_iteration,
    convergence_tol=CONVERGENCE_TOL,
)
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
#post_process(mod_object)
