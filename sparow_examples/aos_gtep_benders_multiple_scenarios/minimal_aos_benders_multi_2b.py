#!/usr/bin/env python3
"""Minimal multi-scenario GTEP + classical Benders + true AOS (Rung 2b).

Adapted from minimal_aos_benders_gtep.py for aos_gtep_benders_multiple_scenarios.

Two scenarios:
  load_0p95  → alpha=0.95, Probability=0.5
  load_1p05  → alpha=1.05, Probability=0.5

Uses SPAROW BendersSolver.solve_and_return_model (no master-capture patch).

Augmentation
------------
After Benders converges, snapshot the master optimum with as_solution /
PyomoSolution *before* generate_candidates (which mutates the master). That
known-feasible solution is added into the candidate pool for the filter so we
always test at least one point the filter has already been shown to accept.

Sizes (Rung 2b)
---------------
  stages=2, num_reps=1, len_reps=12, num_commit=8, num_dispatch=1
"""
from __future__ import annotations

import importlib
import json
import logging
import time

import pyomo.environ as pyo

logging.getLogger("pyomo.core").setLevel(logging.ERROR)

print("--- Loading create_sp and building multi-scenario SP (Rung 2b) ---", flush=True)
t_build = time.perf_counter()
try:
    from sparow_examples.aos_gtep_benders_multiple_scenarios.scale_tests import create_sp
except ImportError:
    from scale_tests import create_sp

sp = create_sp(
    stages=2,
    num_reps=1,
    len_reps=12,
    num_commit=8,
    num_dispatch=1,
)
print(
    f"  SP built in {time.perf_counter() - t_build:.1f}s  "
    f"bundles={list(sp.bundles)}",
    flush=True,
)

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

TIME_LIMIT = 600.0
CONVERGENCE_TOL = 1e-3
MAX_ITERATIONS = 50


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
t_benders = time.perf_counter()
result = solver.solve_and_return_model(
    sp,
    eta_bounds_map,
    subproblem_transforms=[relax_second_stage],
    master_transforms=None,
    on_iteration=_on_iteration,
    convergence_tol=CONVERGENCE_TOL,
)
print(f"  Benders finished in {time.perf_counter() - t_benders:.1f}s", flush=True)

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


def _master_as_solution(m):
    """Pyomo/OR-TOPAS solution from the live master (Benders optimum)."""
    errors = []
    for path in (
        "or_topas.benders.aos_benders",
        "or_topas.solnpool.solnpool",
        "or_topas.aos",
        "or_topas.solnpool.solution",
        "aos_benders",
    ):
        try:
            mod = importlib.import_module(path)
            for attr in ("as_solution", "_as_pyomo_solution"):
                fn = getattr(mod, attr, None)
                if fn is not None:
                    print(f"  {attr} from {path}", flush=True)
                    return fn(m)
        except Exception as e:
            errors.append(f"{path}: {e}")
    try:
        from or_topas.solnpool.solution import PyomoSolution

        vars_list = list(
            m.component_data_objects(pyo.Var, active=True, descend_into=True)
        )
        objs = list(
            m.component_data_objects(pyo.Objective, active=True, descend_into=True)
        )
        print(
            f"  PyomoSolution(variables={len(vars_list)}, objectives={len(objs)})",
            flush=True,
        )
        return PyomoSolution(variables=vars_list, objectives=objs or None)
    except Exception as e:
        errors.append(f"PyomoSolution: {e}")
    raise ImportError(
        "Could not build as_solution / PyomoSolution:\n  " + "\n  ".join(errors)
    )


def _add_to_pool(pool, solution):
    """Add a solution object into an existing pool."""
    if hasattr(pool, "add"):
        try:
            pool.add(solution)
            return
        except TypeError:
            try:
                pool.add(solution=solution)
                return
            except TypeError:
                if hasattr(solution, "variables") and hasattr(solution, "objectives"):
                    pool.add(
                        variables=solution.variables,
                        objectives=solution.objectives,
                    )
                    return
                raise
    raise RuntimeError(f"Pool type {type(pool)} has no usable add(...)")


# ---------------------------------------------------------------------------
# Snapshot Benders optimum BEFORE generate_candidates mutates the master
# ---------------------------------------------------------------------------
print(
    "--- Snapshot Benders optimum (before generate_candidates) ---",
    flush=True,
)
benders_candidate = _master_as_solution(master)
try:
    obj_bc = (
        benders_candidate.objective().value
        if hasattr(benders_candidate, "objective")
        else getattr(benders_candidate, "objective", None)
    )
    print(f"  benders_candidate objective = {obj_bc}", flush=True)
except Exception as e:
    print(f"  (could not read benders_candidate objective: {e})", flush=True)

NUM_SOLUTIONS = 200
REL_GAP = 0.00001          # generate: tight on the master
FILTER_REL_GAP = 0.05   # filter: looser true-cost window (~2%; observed η-gap ~1%)

print("--- aos-benders: generate_candidates ---", flush=True)
t_gen = time.perf_counter()
candidates, data = aos_benders_generate_candidates(
    m=master,
    rel_gap=REL_GAP,
    num_solutions=NUM_SOLUTIONS,
    mip_solver="gurobi",
    enumeration_method="gurobi_pool",
    tee=False,
)
print(
    f"  Candidates from generate: {len(candidates)}  "
    f"({time.perf_counter() - t_gen:.1f}s)",
    flush=True,
)

# Inject known-feasible Benders optimum into the generate pool
print(
    "--- Inject Benders optimum into candidate pool ---",
    flush=True,
)
n_before = len(candidates)
try:
    _add_to_pool(candidates, benders_candidate)
    print(
        f"  Pool size {n_before} → {len(candidates)} "
        f"(added Benders as_solution snapshot)",
        flush=True,
    )
except Exception as e:
    print(f"  Could not add to generate pool ({e}); building combined pool", flush=True)
    # Fallback: new pool of same type with Benders sol + generate sols
    try:
        combined = type(candidates)()
        _add_to_pool(combined, benders_candidate)
        for sol in candidates:
            try:
                _add_to_pool(combined, sol)
            except Exception:
                pass
        candidates = combined
        print(f"  Combined pool size: {len(candidates)}", flush=True)
    except Exception as e2:
        raise RuntimeError(
            f"Failed to inject Benders solution into candidate pool: {e}; {e2}"
        ) from e2

print("--- aos-benders: filter (true solutions) ---", flush=True)
# Separate generate vs filter tolerance: master η underestimates true cost (~1% here).
override_upper = data.lower_bound * (1.0 + FILTER_REL_GAP)
print(
    f"  generate rel_gap={REL_GAP}; filter_rel_gap={FILTER_REL_GAP}\n"
    f"  inclusion LB={data.lower_bound:.6g}; "
    f"override UB={override_upper:.6g} "
    f"(data.upper_bound was {data.upper_bound:.6g})",
    flush=True,
)
t_filter = time.perf_counter()
true_pool = aos_benders_filter(
    candidates,
    data,
    tee=False,
    tee_final=True,
    override_upper_bound=override_upper,
)
print(
    f"  True solutions: {len(true_pool)}  "
    f"({time.perf_counter() - t_filter:.1f}s)",
    flush=True,
)

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

if len(true_pool) == 0:
    print(
        "\n*** WARNING: filter accepted nothing, including injected Benders opt. ***\n"
        "*** Unexpected given the prior diagnostic. Check SPAROW bounds fix / load. ***",
        flush=True,
    )
elif len(true_pool) == 1:
    print(
        "\n*** Only the injected Benders optimum survived; generate pool still empty. ***",
        flush=True,
    )
else:
    print(
        f"\n*** {len(true_pool)} true solutions (Benders opt + generate survivors). ***",
        flush=True,
    )

# ---------------------------------------------------------------------------
# First-stage-only archive (Hamming EF projection pattern)
# Project each true solution onto SPAROW first-stage names so the result is
# compatible with sparow.sp.util.constrain_EF_model(..., first_stage_variables=...).
# ---------------------------------------------------------------------------
FS_OUTPUT_FILE = "aos_benders_multi_2b_true_fs_candidate_gap_1eminus5_filter_gap_5eminus2.json"
print(f"\n--- First-stage archive ---", flush=True)

b0 = next(iter(sp.bundles))
fs_name_set = set()
if hasattr(sp, "int_to_FirstStageVar") and b0 in sp.int_to_FirstStageVar:
    fs_name_set = {v.name for v in sp.int_to_FirstStageVar[b0].values()}
elif hasattr(sp, "int_to_FirstStageVarName") and sp.int_to_FirstStageVarName:
    fs_name_set = set(sp.int_to_FirstStageVarName.values())

if not fs_name_set:
    print(
        "  WARNING: SPAROW first-stage maps empty; falling back to name patterns",
        flush=True,
    )

    def is_first_stage(name: str) -> bool:
        n = (name or "").lower()
        return ("investmentstage" in n) and (
            "binary_indicator" in n or "renewable" in n
        )
else:
    def is_first_stage(name: str) -> bool:
        return name in fs_name_set

print(f"  First-stage name set size: {len(fs_name_set)}", flush=True)
if fs_name_set:
    for n in sorted(fs_name_set)[:6]:
        print(f"    {n}", flush=True)
    if len(fs_name_set) > 6:
        print(f"    ... +{len(fs_name_set) - 6} more", flush=True)


def extract_first_stage(sol):
    """Project a Solution onto the first-stage variables.

    Returns a dict compatible with
    sparow.sp.util.constrain_EF_model(..., first_stage_variables=...).
    """
    fs = {}
    try:
        var_iter = sol.variables()
    except TypeError:
        var_iter = sol.variables
    for vinfo in var_iter:
        name = getattr(vinfo, "name", None)
        if name and is_first_stage(name):
            val = getattr(vinfo, "value", None)
            fs[name] = float(val) if val is not None else None
    obj_val = None
    try:
        if hasattr(sol, "objectives") and callable(sol.objectives):
            objs = list(sol.objectives())
            if objs:
                obj_val = float(objs[0].value)
        elif hasattr(sol, "objective") and callable(sol.objective):
            obj_val = float(sol.objective().value)
    except Exception:
        pass
    return {
        "id": getattr(sol, "id", None),
        "objective": obj_val,
        "first_stage_variables": fs,
    }


fs_archive = {
    "metadata": {
        "description": (
            "First-stage projection of aos-benders true solutions "
            "(multi-scenario Rung 2b, investment indicators only)"
        ),
        "source_script": "minimal_aos_benders_multi_2b.py",
        "num_solutions": len(true_pool),
        "generate_rel_gap": REL_GAP,
        "filter_rel_gap": FILTER_REL_GAP,
        "first_stage_name_set_size": len(fs_name_set),
        "size": {
            "stages": 2,
            "num_reps": 1,
            "len_reps": 12,
            "num_commit": 8,
            "num_dispatch": 1,
        },
        "scenarios": [
            {"ID": "load_0p95", "alpha": 0.95, "Probability": 0.5},
            {"ID": "load_1p05", "alpha": 1.05, "Probability": 0.5},
        ],
    },
    "solutions": [extract_first_stage(s) for s in true_pool],
}

if fs_archive["solutions"]:
    n_fs = len(fs_archive["solutions"][0]["first_stage_variables"])
    fs_archive["metadata"]["first_stage_vars_retained_per_sol"] = n_fs
    print(f"  Retained {n_fs} first-stage variables per solution", flush=True)
    if n_fs == 0:
        print(
            "  WARNING: zero first-stage variables matched. "
            "Inspect fs_name_set vs. VariableInfo.name values "
            "(master compact_repn prefixes may differ from SP maps).",
            flush=True,
        )
    for i, rec in enumerate(fs_archive["solutions"]):
        print(f"  Sol {i}: objective={rec['objective']}, n_fs={len(rec['first_stage_variables'])}", flush=True)
else:
    print("  WARNING: no solutions to project", flush=True)

with open(FS_OUTPUT_FILE, "w") as f:
    json.dump(fs_archive, f, indent=2)
print(f"  Written to {FS_OUTPUT_FILE}", flush=True)

print("Done.", flush=True)
print(
    f"Phase times (s): wall={time.perf_counter() - t_build:.1f} "
    f"(see per-phase lines above)",
    flush=True,
)
