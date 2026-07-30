#!/usr/bin/env python3
"""
run_aos_analysis_tiny_uc.py
===========================

AOS analysis on the synthetic tiny UC (EGRET + SPAROW) using the Gurobi
Solution Pool method for both:

  Path A  – AOS on the Extensive Form
  Path B  – AOS-Benders (gurobi_pool as the candidate-generation kernel)

Residual second-stage discretes are handled by the official SPAROW utility::

    from sparow.sp.util import relax_second_stage

exactly as in gtep_sparow_staged.py / aos_single_ef.py:

  - EF path (optional): sp.add_transformation(relax_second_stage)
  - Benders path:       solver.solve(..., subproblem_transforms=[relax_second_stage],
                                            master_transforms=None)

Master deliberately does NOT receive residual relaxation so first-stage
binaries stay discrete.

Assumptions
-----------
- Full stack installed: pyomo, egret, sparow, or_topas, gurobi
- Attachment modules on PYTHONPATH or next to this script

Usage
-----
  python run_aos_analysis_tiny_uc.py
  python run_aos_analysis_tiny_uc.py --n-periods 4 --num-sols 12 --rel-gap 0.0
  python run_aos_analysis_tiny_uc.py --rel-gap 0.05
  python run_aos_analysis_tiny_uc.py --skip-farmer-kernel
  python run_aos_analysis_tiny_uc.py --skip-ef
  python run_aos_analysis_tiny_uc.py --skip-egret-benders
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyomo.environ as pyo

# ---------------------------------------------------------------------------
# Make attachment / local modules importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_PATHS = [
    _SCRIPT_DIR,
    os.path.join(_SCRIPT_DIR, "aos_analysis"),
    os.path.join(os.path.dirname(_SCRIPT_DIR), "attachments"),
    "/home/workdir/attachments",
    "/home/workdir/artifacts/aos_analysis",
]
for p in _CANDIDATE_PATHS:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("aos_tiny_uc")


def _print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Pool pretty-printing (or_topas solnpool API)
# ---------------------------------------------------------------------------

def _sol_objective(sol) -> Optional[float]:
    try:
        if hasattr(sol, "objective") and callable(sol.objective):
            obj = sol.objective()
            if hasattr(obj, "value"):
                return float(obj.value)
            return float(obj)
        if hasattr(sol, "objective") and sol.objective is not None and not callable(sol.objective):
            o = sol.objective
            return float(o.value if hasattr(o, "value") else o)
        if hasattr(sol, "objectives") and sol.objectives:
            o = sol.objectives[0]
            return float(o.value if hasattr(o, "value") else o)
    except Exception:
        pass
    return None


def _extract_uniton_pattern(sol_or_model, max_items: int = 20) -> List[Tuple[str, float]]:
    pairs: List[Tuple[str, float]] = []
    try:
        vars_list = None
        if hasattr(sol_or_model, "_variables"):
            vars_list = sol_or_model._variables
        elif hasattr(sol_or_model, "variables") and not callable(sol_or_model.variables):
            vars_list = sol_or_model.variables

        if vars_list is not None:
            for v in vars_list:
                name = getattr(v, "name", None) or str(v)
                if "uniton" not in name.lower():
                    continue
                val = getattr(v, "value", None)
                try:
                    pairs.append((name, float(val) if val is not None else float("nan")))
                except Exception:
                    pairs.append((name, float("nan")))
            return pairs[:max_items]

        if hasattr(sol_or_model, "component_data_objects"):
            for v in sol_or_model.component_data_objects(
                pyo.Var, active=True, descend_into=True
            ):
                if "uniton" in v.name.lower():
                    try:
                        pairs.append((v.name, float(pyo.value(v))))
                    except Exception:
                        pairs.append((v.name, float("nan")))
            return pairs[:max_items]
    except Exception as e:
        pairs.append((f"<extract failed: {e}>", float("nan")))
    return pairs[:max_items]


def _summarize_pool(pool, title: str = "Pool", max_sols: int = 12):
    _print(f"\n  --- {title} (showing up to {max_sols} solutions) ---")
    try:
        n = len(pool)
    except TypeError:
        try:
            n = pool.size() if callable(getattr(pool, "size", None)) else "?"
        except Exception:
            n = "?"
    _print(f"  size = {n}")

    try:
        iterable = list(pool)
    except TypeError:
        iterable = getattr(pool, "solutions", None) or getattr(pool, "_solutions", None) or []
        if hasattr(iterable, "values"):
            iterable = list(iterable.values())
        else:
            iterable = list(iterable) if iterable else []

    for i, sol in enumerate(iterable[:max_sols]):
        obj = _sol_objective(sol)
        pattern = _extract_uniton_pattern(sol)
        compact = tuple(int(round(v)) if v == v else -1 for _, v in pattern)
        _print(f"  [{i:2d}] obj={obj}  UnitOn={compact}")
        if pattern and len(pattern) <= 12:
            _print(f"         detail: {pattern}")


# ---------------------------------------------------------------------------
# Shared: build the synthetic tiny-UC SPAROW SP
# ---------------------------------------------------------------------------

def _build_tiny_uc_sp(
    n_periods: int = 4,
    first_stage_names: Optional[Sequence[str]] = None,
):
    from egret_sparow_staged import (
        stage0_check_environment,
        stage1_build_egret_model,
        stage2_inspect_variables,
        stage3_verify_first_stage_names,
        stage5_egret_sparow_sp,
    )

    stage0_check_environment()
    md, model = stage1_build_egret_model(force_synthetic=True, n_periods=n_periods)
    binary_comps, _, suggested = stage2_inspect_variables(model)
    if first_stage_names is None:
        first_stage_names = [
            b["name"] for b in binary_comps if "uniton" in b["name"].lower()
        ] or suggested
    stage3_verify_first_stage_names(model, first_stage_names)
    _print(f"  first_stage_names = {list(first_stage_names)}")

    sp = stage5_egret_sparow_sp(
        md,
        first_stage_names=first_stage_names,
        scenario_scales={"base": 1.0},
    )
    return sp, list(first_stage_names)


def _seed_and_unfix_first_stage(sp) -> Tuple[int, int]:
    """
    Seed / unfix first-stage vars so the standard_lp transform never sees
    value=None and pure first-stage constraints do not collapse to a Python
    bool when some indicators were fixed from initial conditions.
    """
    n_seeded = 0
    n_unfixed = 0
    for b in sp.bundles:
        fs_map = getattr(sp, "int_to_FirstStageVar", {}).get(b, {})
        for v in fs_map.values():
            if v.fixed:
                val = v.value
                v.unfix()
                if val is not None:
                    if v.lb is None or val > v.lb:
                        v.setlb(val)
                    if v.ub is None or val < v.ub:
                        v.setub(val)
                    v.set_value(val, skip_validation=True)
                n_unfixed += 1
            if v.value is None:
                if v.lb is not None:
                    v.set_value(v.lb, skip_validation=True)
                elif v.ub is not None:
                    v.set_value(v.ub, skip_validation=True)
                else:
                    v.set_value(0.0, skip_validation=True)
                n_seeded += 1
    return n_seeded, n_unfixed


# ---------------------------------------------------------------------------
# Path A: Extensive-form AOS with Gurobi pool
# ---------------------------------------------------------------------------

def run_ef_aos(
    n_periods: int = 4,
    first_stage_names: Optional[Sequence[str]] = None,
    num_solutions: int = 12,
    rel_opt_gap: float = 0.0,
    also_positive_gap: float = 0.05,
    tee: bool = False,
) -> Dict[str, Any]:
    """
    Build synthetic tiny UC SP, obtain the EF model, enumerate near-optimal
    solutions with Gurobi Solution Pool (PoolSearchMode=2).

    For pure MIP AOS on the EF we deliberately do *not* register
    relax_second_stage — the discrete structure is what the pool enumerates.
    """
    _print("\n" + "=" * 72)
    _print("PATH A: Extensive-Form AOS (Gurobi pool) on synthetic tiny UC")
    _print("=" * 72)

    from egret_sparow_staged import stage6_solve_extensive_form
    from sparow.ef import ExtensiveFormSolver

    sp, fs_names = _build_tiny_uc_sp(n_periods, first_stage_names)

    ef_ref = stage6_solve_extensive_form(sp, solver_name="gurobi", time_limit=180)
    _print(f"  EF reference objective = {ef_ref['objective']:.6g}")

    ef_solver = ExtensiveFormSolver()
    ef_solver.set_options(solver="gurobi", solver_options={"TimeLimit": 180})
    res = ef_solver.solve_and_return_EF(sp)
    M = res.model
    _print(f"  EF model type: {type(M)}")
    objs = list(M.component_data_objects(pyo.Objective, active=True, descend_into=False))
    _print(f"  Active top-level objectives: {[o.name for o in objs]}")

    try:
        from or_topas.aos import gurobi_generate_solutions
    except ImportError:
        from gurobi_solnpool import gurobi_generate_solutions

    gaps_to_run = [rel_opt_gap]
    if also_positive_gap and also_positive_gap != rel_opt_gap:
        gaps_to_run.append(also_positive_gap)

    pools = {}
    for gap in gaps_to_run:
        _print(f"\n--- Gurobi Solution Pool AOS on EF (rel_opt_gap={gap}) ---")
        res_g = ef_solver.solve_and_return_EF(sp)
        Mg = res_g.model
        pool = gurobi_generate_solutions(
            model=Mg,
            num_solutions=num_solutions,
            rel_opt_gap=gap,
            pool_search_mode=2,
            tee=tee,
        )
        _summarize_pool(pool, title=f"EF AOS pool (rel_gap={gap})")
        pools[gap] = pool

    return {
        "sp": sp,
        "ef_model": M,
        "ef_reference": ef_ref,
        "aos_pools": pools,
        "first_stage_names": fs_names,
    }


# ---------------------------------------------------------------------------
# Path B: AOS-Benders with gurobi_pool on the EGRET SP
# ---------------------------------------------------------------------------

def _capture_upper_model_from_sparow_benders(
    sp,
    ef_reference: dict,
    max_iterations: int = 40,
    eta_lower: float = 0.0,
) -> Any:
    """
    Run SPAROW BendersSolver with the official residual-discrete handling:

        from sparow.sp.util import relax_second_stage
        solver.solve(..., subproblem_transforms=[relax_second_stage],
                          master_transforms=None)

    Master does not receive residual relaxation (first-stage stay discrete).
    Captures the instrumented upper_model for aos_benders.
    """
    from sparow.benders import BendersSolver
    from sparow.sp.util import relax_second_stage

    ef_obj = ef_reference["objective"]
    eta_bounds_map = {b: (float(eta_lower), None) for b in sp.bundles}
    for b, bounds in eta_bounds_map.items():
        if bounds[0] is None:
            raise RuntimeError(f"eta lower bound for bundle {b!r} is None")

    n_seeded, n_unfixed = _seed_and_unfix_first_stage(sp)
    _print(f"  first-stage: seeded={n_seeded}, unfixed={n_unfixed}")

    # Capture upper_model after the master transform
    captured = {"upper_model": None}
    original_transform = BendersSolver._transform_to_master_model

    @staticmethod
    def _capturing_transform(**kwargs):
        m = original_transform(**kwargs)
        captured["upper_model"] = m
        return m

    BendersSolver._transform_to_master_model = _capturing_transform

    solver = BendersSolver()
    # Both master and subproblem must be persistent (plain "gurobi" → GUROBIFILE
    # has no set_instance).
    solver.set_options(
        solver="gurobi_persistent",
        subproblem_solver="gurobi_persistent",
        max_iterations=max_iterations,
        is_persistent_solver=True,
        allow_infeasible_subproblems=True,
        loglevel="INFO",
    )
    for attr in ("allow_infeasible_subproblems", "allow_infeasible"):
        if hasattr(solver, attr):
            setattr(solver, attr, True)

    _print(
        "  Calling BendersSolver.solve(\n"
        "      subproblem_transforms=[relax_second_stage],\n"
        "      master_transforms=None,\n"
        "      gurobi_persistent, allow_infeasible_subproblems=True ...\n"
        "  )"
    )
    try:
        results = solver.solve(
            sp,
            eta_bounds_map=eta_bounds_map,
            error_on_initialized_root_vars=False,
            convergence_tol=1e-8,
            subproblem_transforms=[relax_second_stage],
            master_transforms=None,
        )
    finally:
        BendersSolver._transform_to_master_model = original_transform

    upper = captured["upper_model"]
    if upper is None:
        raise RuntimeError(
            "Failed to capture upper_model from SPAROW BendersSolver. "
            "Internal call sequence may have changed."
        )

    benders_blocks = [
        b
        for b in upper.component_data_objects(pyo.Block, descend_into=True)
        if "BendersGenerator" in str(type(b))
    ]
    _print(f"  Captured master; BendersGenerator blocks found: {len(benders_blocks)}")
    if not benders_blocks and not hasattr(upper, "benders"):
        raise RuntimeError(
            "Captured master has no BendersGenerator block – "
            "aos_benders_generate_candidates will refuse to run."
        )

    try:
        benders_obj = float(pyo.value(upper.obj))
        _print(f"  Benders master objective after cut loop: {benders_obj:.6g}")
        _print(
            f"  EF reference: {ef_obj:.6g}  |diff| = {abs(benders_obj - ef_obj):.4g}"
        )
    except Exception as e:
        _print(f"  (Could not evaluate master objective: {e})")

    return upper, results


def run_aos_benders_egret(
    n_periods: int = 4,
    first_stage_names: Optional[Sequence[str]] = None,
    num_solutions: int = 12,
    rel_opt_gap: float = 0.0,
    max_benders_iterations: int = 40,
    tee: bool = False,
) -> Dict[str, Any]:
    """
    Full Path B on the synthetic tiny UC:

      1. Build SP + EF reference
      2. Run SPAROW Benders with subproblem_transforms=[relax_second_stage]
      3. Capture master → aos_benders_generate_candidates(..., "gurobi_pool")
      4. aos_benders_filter → report true near-optimal pool
    """
    _print("\n" + "=" * 72)
    _print("PATH B: AOS-Benders (gurobi_pool kernel) on synthetic tiny UC")
    _print("=" * 72)

    from egret_sparow_staged import stage6_solve_extensive_form

    sp, fs_names = _build_tiny_uc_sp(n_periods, first_stage_names)
    ef_ref = stage6_solve_extensive_form(sp, solver_name="gurobi", time_limit=180)
    _print(f"  EF reference objective = {ef_ref['objective']:.6g}")

    try:
        upper_model, benders_results = _capture_upper_model_from_sparow_benders(
            sp,
            ef_reference=ef_ref,
            max_iterations=max_benders_iterations,
        )
    except Exception as e:
        _print(f"\n  *** Capture / Benders failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "sp": sp, "ef_reference": ef_ref}

    try:
        from aos_benders import (
            aos_benders_generate_candidates,
            aos_benders_filter,
        )
    except ImportError:
        from or_topas.benders.aos_benders import (
            aos_benders_generate_candidates,
            aos_benders_filter,
        )

    _print("\n--- aos_benders_generate_candidates (enumeration_method='gurobi_pool') ---")
    candidate_pool, data = aos_benders_generate_candidates(
        m=upper_model,
        rel_gap=rel_opt_gap,
        num_solutions=num_solutions,
        mip_solver="gurobi",
        enumeration_method="gurobi_pool",
        tee=tee,
    )
    _summarize_pool(candidate_pool, title="Candidate pool (before filter)")

    _print("\n--- aos_benders_filter ---")
    true_pool = aos_benders_filter(candidate_pool, data, tee=tee, tee_final=True)
    _summarize_pool(true_pool, title=f"True AOS-Benders pool (rel_gap={rel_opt_gap})")

    return {
        "sp": sp,
        "ef_reference": ef_ref,
        "upper_model": upper_model,
        "candidate_pool": candidate_pool,
        "true_pool": true_pool,
        "first_stage_names": fs_names,
    }


# ---------------------------------------------------------------------------
# Kernel sanity-check on Farmer (independent of EGRET)
# ---------------------------------------------------------------------------

def run_aos_benders_kernel_demo(
    num_solutions: int = 10,
    rel_opt_gap: float = 0.01,
    tee: bool = False,
):
    """
    Validate the AOS-Benders generation kernel on Farmer with
    enumeration_method="gurobi_pool".

    Must pass "gurobi_persistent" so setup_farmer_persistent gets a solver
    that implements set_instance (plain "gurobi" → GUROBIFILE → AttributeError).
    """
    _print("\n" + "=" * 72)
    _print("KERNEL DEMO: AOS-Benders with gurobi_pool on Farmer")
    _print("=" * 72)

    try:
        from aos_benders import (
            aos_benders_generate_candidates,
            aos_benders_filter,
        )
    except ImportError:
        from or_topas.benders.aos_benders import (
            aos_benders_generate_candidates,
            aos_benders_filter,
        )

    try:
        import or_topas.benders.tests.test_cases as tc
    except ImportError:
        import test_cases as tc

    opt, m = tc.Farmer.run_farmer(
        "gurobi_persistent",
        mode="s",
        transform="standard_lp",
        add_upper_bounds=True,
        is_persistent=True,
    )

    candidate_pool, data = aos_benders_generate_candidates(
        m=m,
        rel_gap=rel_opt_gap,
        num_solutions=num_solutions,
        mip_solver="gurobi",
        enumeration_method="gurobi_pool",
        tee=tee,
    )
    true_pool = aos_benders_filter(candidate_pool, data, tee=tee, tee_final=True)
    _print(f"  True (filtered) AOS-Benders pool size: {len(true_pool)}")
    _summarize_pool(true_pool, title="Farmer true AOS-Benders pool")
    return true_pool


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AOS analysis (EF + AOS-Benders) on synthetic tiny UC"
    )
    parser.add_argument("--n-periods", type=int, default=4)
    parser.add_argument("--num-sols", type=int, default=12)
    parser.add_argument(
        "--rel-gap",
        type=float,
        default=0.0,
        help="Relative optimality gap for the pool / AOS filter",
    )
    parser.add_argument(
        "--also-gap",
        type=float,
        default=0.05,
        help="Also run EF-AOS at this positive gap (0 to disable)",
    )
    parser.add_argument("--max-benders-iter", type=int, default=40)
    parser.add_argument("--skip-ef", action="store_true")
    parser.add_argument("--skip-egret-benders", action="store_true")
    parser.add_argument("--skip-farmer-kernel", action="store_true")
    parser.add_argument("--tee", action="store_true", default=False)
    args = parser.parse_args()

    results = {}

    if not args.skip_ef:
        results["ef_aos"] = run_ef_aos(
            n_periods=args.n_periods,
            num_solutions=args.num_sols,
            rel_opt_gap=args.rel_gap,
            also_positive_gap=args.also_gap,
            tee=args.tee,
        )

    if not args.skip_egret_benders:
        results["egret_aos_benders"] = run_aos_benders_egret(
            n_periods=args.n_periods,
            num_solutions=args.num_sols,
            rel_opt_gap=args.rel_gap,
            max_benders_iterations=args.max_benders_iter,
            tee=args.tee,
        )

    if not args.skip_farmer_kernel:
        results["farmer_aos_benders"] = run_aos_benders_kernel_demo(
            num_solutions=args.num_sols,
            rel_opt_gap=max(args.rel_gap, 0.01),
            tee=False,
        )

    _print("\n" + "=" * 72)
    _print("*** Finished. Inspect the pools printed above. ***")
    _print("=" * 72)
    return results


if __name__ == "__main__":
    main()
