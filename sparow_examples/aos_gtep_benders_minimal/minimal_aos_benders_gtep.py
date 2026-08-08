#!/usr/bin/env python3
"""Minimal GTEP + classical Benders + true AOS on aos_single.

Uses SPAROW BendersSolver.solve_and_return_model (no master-capture patch).
No CLI. Hard-coded aos_single.
"""
from __future__ import annotations
import time

print("--- Loading create_sp and building SP ---", flush=True)
t0 = time.perf_counter()
from sparow_examples.aos_gtep_9bus.aos_single import create_sp
sp = create_sp()
print(f"  SP built in {time.perf_counter()-t0:.1f}s  bundles={list(sp.bundles)}", flush=True)

# seed / unfix first-stage
n_unfixed = n_seeded = 0
for b in sp.bundles:
    for v in getattr(sp, "int_to_FirstStageVar", {}).get(b, {}).values():
        if getattr(v, "fixed", False):
            val = v.value
            v.unfix()
            if val is not None:
                try:
                    v.set_value(val, skip_validation=True)
                except Exception:
                    pass
            n_unfixed += 1
        if v.value is None:
            v.set_value(v.lb if v.lb is not None else 0.0, skip_validation=True)
            n_seeded += 1
print(f"  first-stage: unfixed={n_unfixed}, seeded={n_seeded}", flush=True)

from sparow.benders import BendersSolver
from sparow.sp.util import relax_second_stage

eta_bounds_map = {b: (-1e6, None) for b in sp.bundles}
print(f"  eta_bounds_map: {eta_bounds_map}", flush=True)

TIME_LIMIT = 240.0  # seconds; applied if SPAROW forwards solver_options

def _on_iteration(data):
    n_cuts = len(data.cuts_added) if data.cuts_added is not None else 0
    print(
        f"  --- Benders iteration {data.iter_idx}  cuts_added={n_cuts} ---",
        flush=True,
    )

solver = BendersSolver()
solver.set_options(
    solver="gurobi_persistent",
    subproblem_solver="gurobi_persistent",
    max_iterations=50,
    is_persistent_solver=True,
    allow_infeasible_subproblems=True,
    loglevel="INFO",
    solver_options={"TimeLimit": TIME_LIMIT, "timelimit": TIME_LIMIT},
)

print(
    f"--- Calling solve_and_return_model (TimeLimit={TIME_LIMIT}s) ---",
    flush=True,
)
t0 = time.perf_counter()
result = solver.solve_and_return_model(
    sp,
    eta_bounds_map,
    subproblem_transforms=[relax_second_stage],
    master_transforms=None,
    on_iteration=_on_iteration,
)
print(f"  Benders finished in {time.perf_counter()-t0:.1f}s", flush=True)

master = result.upper_model
if master is None:
    raise RuntimeError("solve_and_return_model did not return upper_model")
print(f"  Master recovered via solve_and_return_model (type={type(master)})", flush=True)

try:
    from aos_benders import aos_benders_generate_candidates, aos_benders_filter
except ImportError:
    from or_topas.benders.aos_benders import (
        aos_benders_generate_candidates,
        aos_benders_filter,
    )

print("--- aos-benders: generate_candidates ---", flush=True)
t0 = time.perf_counter()
candidates, data = aos_benders_generate_candidates(
    m=master,
    rel_gap=0.0,
    num_solutions=3,
    mip_solver="gurobi",
    enumeration_method="gurobi_pool",
    tee=False,
)
print(f"  Candidates: {len(candidates)}  ({time.perf_counter()-t0:.1f}s)", flush=True)

print("--- aos-benders: filter (true solutions) ---", flush=True)
t0 = time.perf_counter()
true_pool = aos_benders_filter(candidates, data, tee=False, tee_final=False)
print(f"  True solutions: {len(true_pool)}  ({time.perf_counter()-t0:.1f}s)", flush=True)

for i, sol in enumerate(true_pool):
    try:
        obj = (
            sol.objective().value
            if hasattr(sol, "objective")
            else getattr(sol, "objective", None)
        )
        print(f"  Sol {i}: objective = {obj}", flush=True)
    except Exception as e:
        print(f"  Sol {i}: (could not read objective: {e})", flush=True)

print("Done.", flush=True)
