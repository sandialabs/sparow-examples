#!/usr/bin/env python3
"""Diagnostic: filter the known-feasible Benders optimum using aos-benders data.

CRITICAL ORDER
--------------
Snapshot the Benders optimum from the master *immediately* after
solve_and_return_model — *before* aos_benders_generate_candidates.

generate_candidates mutates the master (pool search loads other points).
as_solution(master) after generate would test the *last pool point*, not the
known Benders optimum, and would invalidate the diagnostic.

Flow
----
1. Classical Benders → master holds the optimal point
2. Census fixed-at-0 / lb>0 on master
3. as_solution / PyomoSolution on master → wrap in a pool  (Benders optimum)
4. generate_candidates on master → keep `data` only; ignore candidates
   (master may be mutated after this; we already have the Benders pool)
5. aos_benders_filter(benders_pool, data)

Sizes (Rung 2b)
---------------
  stages=2, num_reps=1, len_reps=12, num_commit=8, num_dispatch=1
"""
from __future__ import annotations

import importlib
import logging
import time

import pyomo.environ as pyo

logging.getLogger("pyomo.core").setLevel(logging.ERROR)

print("--- Loading create_sp and building SP (Rung 2b) ---", flush=True)
t_build = time.perf_counter()
try:
    from sparow_examples.aos_gtep_benders_minimal.scale_tests import create_sp
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
print(f"  Master recovered (type={type(master)})", flush=True)

# Census: vars fixed at 0 with lb > 0 (should be ~0 after SPAROW bounds fix)
n_fixed0 = 0
n_fixed0_oob = 0
oob_samples = []
for v in master.component_data_objects(pyo.Var, active=True, descend_into=True):
    if not getattr(v, "fixed", False):
        continue
    try:
        val = pyo.value(v)
    except Exception:
        continue
    if val is None or abs(float(val)) > 1e-12:
        continue
    n_fixed0 += 1
    lb = v.lb
    if lb is not None and float(lb) > 0:
        n_fixed0_oob += 1
        if len(oob_samples) < 8:
            oob_samples.append((v.name, float(lb), v.ub))
print(
    f"  Master fixed-at-0 vars: {n_fixed0}; "
    f"of which lb>0 (out-of-bound fix): {n_fixed0_oob}",
    flush=True,
)
for name, lb, ub in oob_samples:
    print(f"    sample OOB fix: {name}  lb={lb} ub={ub} fixed_at=0", flush=True)

try:
    from aos_benders import aos_benders_generate_candidates, aos_benders_filter
except ImportError:
    from or_topas.benders.aos_benders import (
        aos_benders_generate_candidates,
        aos_benders_filter,
    )


def _master_as_solution(m):
    """Pyomo/OR-TOPAS solution object from the live master point."""
    errors = []
    for path in (
        "or_topas.aos",
        "or_topas.benders.aos_benders",
        "or_topas.solnpool.solution",
        "or_topas.solnpool.solnpool",
        "or_topas.util.solution",
        "or_topas.util.pyomo_utils",
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
        errors.append(f"PyomoSolution construct: {e}")

    raise ImportError(
        "Could not build as_solution / PyomoSolution for master. Tried:\n  "
        + "\n  ".join(errors)
    )


def _wrap_in_pool(solution, template_pool=None):
    """Wrap a single solution in a pool (filter expects a pool, not a list)."""
    errors = []

    if template_pool is not None:
        try:
            pool = type(template_pool)()
            if hasattr(pool, "add"):
                try:
                    pool.add(solution)
                except TypeError:
                    pool.add(solution=solution)
                print(
                    f"  pool via type(template)={type(template_pool).__name__}",
                    flush=True,
                )
                return pool
        except Exception as e:
            errors.append(f"type(template_pool)(): {e}")

    for path, cls_name in (
        ("sparow.solnpool", "SparowPoolManager"),
        ("or_topas.solnpool", "PoolManager"),
        ("or_topas.solnpool.solnpool", "PoolManager"),
        ("or_topas.solnpool.pool", "SolutionPool"),
        ("or_topas.aos", "SolutionPool"),
    ):
        try:
            mod = importlib.import_module(path)
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue
            pool = cls()
            if hasattr(pool, "add"):
                try:
                    pool.add(solution)
                except TypeError:
                    try:
                        pool.add(solution=solution)
                    except TypeError:
                        if hasattr(solution, "variables") and hasattr(
                            solution, "objectives"
                        ):
                            pool.add(
                                variables=solution.variables,
                                objectives=solution.objectives,
                            )
                        else:
                            raise
                print(f"  pool via {path}.{cls_name}", flush=True)
                return pool
        except Exception as e:
            errors.append(f"{path}.{cls_name}: {e}")

    raise RuntimeError(
        "Could not wrap Benders solution in a pool. Tried:\n  "
        + "\n  ".join(errors)
    )


# ---------------------------------------------------------------------------
# CRITICAL: snapshot Benders optimum BEFORE generate_candidates mutates master
# ---------------------------------------------------------------------------
print(
    "--- Snapshot Benders optimum from master (BEFORE generate_candidates) ---",
    flush=True,
)
benders_candidate = _master_as_solution(master)
print(f"  type(benders_candidate)={type(benders_candidate)}", flush=True)
try:
    obj_bc = (
        benders_candidate.objective().value
        if hasattr(benders_candidate, "objective")
        else getattr(benders_candidate, "objective", None)
    )
    print(f"  benders_candidate objective = {obj_bc}", flush=True)
except Exception as e:
    print(f"  (could not read benders_candidate objective: {e})", flush=True)

# Try to wrap without generate template first; re-wrap after generate if needed
try:
    benders_pool = _wrap_in_pool(benders_candidate, template_pool=None)
    print(f"  type(benders_pool)={type(benders_pool)}, len={len(benders_pool)}", flush=True)
except Exception as e:
    print(f"  Early pool wrap deferred until after generate: {e}", flush=True)
    benders_pool = None

NUM_SOLUTIONS = 20
REL_GAP = 0.01

print(
    "--- aos-benders: generate_candidates (for data only; candidates ignored) ---",
    flush=True,
)
print(
    "  NOTE: generate may mutate master; Benders snapshot already taken above.",
    flush=True,
)
t_gen = time.perf_counter()
candidates_gen, data = aos_benders_generate_candidates(
    m=master,
    rel_gap=REL_GAP,
    num_solutions=NUM_SOLUTIONS,
    mip_solver="gurobi",
    enumeration_method="gurobi_pool",
    tee=False,
)
print(
    f"  generate pool type={type(candidates_gen)}, "
    f"len={len(candidates_gen)}  ({time.perf_counter() - t_gen:.1f}s) "
    f"[candidates ignored for filter input]",
    flush=True,
)
print(f"  type(data)={type(data)}", flush=True)

# If early pool wrap failed, wrap now using generate pool as type template
if benders_pool is None:
    print("--- Wrap Benders solution in pool (using generate pool as type template) ---", flush=True)
    benders_pool = _wrap_in_pool(benders_candidate, template_pool=candidates_gen)
    print(f"  type(benders_pool)={type(benders_pool)}, len={len(benders_pool)}", flush=True)
else:
    # Prefer same type as generate if different
    if type(benders_pool) is not type(candidates_gen):
        print(
            f"  Re-wrapping to match generate pool type "
            f"({type(benders_pool)} → {type(candidates_gen)})",
            flush=True,
        )
        try:
            benders_pool = _wrap_in_pool(benders_candidate, template_pool=candidates_gen)
            print(
                f"  type(benders_pool)={type(benders_pool)}, len={len(benders_pool)}",
                flush=True,
            )
        except Exception as e:
            print(f"  Re-wrap failed, keeping prior pool: {e}", flush=True)

print(
    "--- aos-benders: filter(benders_pool, data from generate) ---",
    flush=True,
)
print(
    "  (Known-feasible Benders optimum should be accepted if filter is correct)",
    flush=True,
)
t_filter = time.perf_counter()
true_pool = aos_benders_filter(
    benders_pool,
    data,
    tee=True,
    tee_final=True,
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
        "\n*** DIAGNOSTIC: known Benders optimum was REJECTED by the filter. ***\n"
        "*** Filter/setup is the problem (e.g. full-SS snapshot / unfix). ***\n"
        "*** Not a failure of the investment solution itself. ***",
        flush=True,
    )
else:
    print(
        "\n*** DIAGNOSTIC: known Benders optimum was ACCEPTED by the filter. ***\n"
        "*** Filter can accept a true solution; investigate generate candidates. ***",
        flush=True,
    )

print("Done.", flush=True)
print(
    f"Phase times (s): wall since start={time.perf_counter() - t_build:.1f} "
    f"(see per-phase lines above)",
    flush=True,
)
