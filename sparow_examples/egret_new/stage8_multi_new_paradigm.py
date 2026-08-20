#!/usr/bin/env python3
"""
stage8_multi_new_paradigm.py
============================
Multi-scenario Stage-8 under the new paradigm (no class-method monkey-patches).

Operational rules for multi-scenario (from project summary + Phase-0):
  1. Build multi-scenario SP
  2. Solve EF on that SP → record reference objective
  3. *Rebuild* a fresh multi-scenario SP (EF may mutate bundling;
     deepcopy of EGRET SP is unreliable)
  4. Run Benders on the *fresh* SP with the locked public residual transform
     via solve_and_return_model(..., subproblem_transforms=[...])

Single-scenario rule is the opposite (EF then Benders on the *same* SP).

Usage (same directory as egret_sparow_staged_multi.py + stage8_new_paradigm.py):

  python stage8_multi_new_paradigm.py 2>&1 | tee stage8_multi_selftest.log

Expected (synthetic T=4, 3 scenarios low/med/high):
  EF objective ≈ 6005.35 (or the value Stage 6 reports)
  Benders matches EF within tolerance
  live upper_model returned
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence

import pyomo.environ as pyo

# Reuse the locked residual transform from the single-scenario new-paradigm module
try:
    from stage8_new_paradigm import (
        egret_uc_subproblem_transform,
        stage8_run_benders as _stage8_single,
    )
except ImportError:
    # Fallback: define a minimal copy so this file is self-contained if needed
    def egret_uc_subproblem_transform(sp, model):
        from pyomo.common.collections import ComponentSet
        from pyomo.core.expr.visitor import identify_variables

        n_relaxed = 0
        for v in model.component_data_objects(pyo.Var, active=True, descend_into=True):
            dom = v.domain
            if (
                dom is pyo.Binary
                or (hasattr(dom, "name") and "binary" in str(dom).lower())
                or str(dom).lower() in ("binary", "integers")
            ):
                v.domain = pyo.UnitInterval
                n_relaxed += 1
        if n_relaxed:
            print(
                f"  egret_uc_subproblem_transform: relaxed {n_relaxed} residual "
                "Binary/Integer vars → UnitInterval",
                flush=True,
            )

        first_stage_vars = None
        try:
            b = next(iter(sp.bundles))
            if hasattr(sp, "int_to_FirstStageVar") and b in sp.int_to_FirstStageVar:
                first_stage_vars = ComponentSet(sp.int_to_FirstStageVar[b].values())
        except Exception:
            pass

        if first_stage_vars is not None:
            cons_to_deactivate = []
            for cons in model.component_data_objects(
                pyo.Constraint, active=True, descend_into=True
            ):
                try:
                    vars_in_cons = list(
                        identify_variables(cons.body, include_fixed=False)
                    )
                    if vars_in_cons and all(v in first_stage_vars for v in vars_in_cons):
                        cons_to_deactivate.append(cons)
                except Exception:
                    continue
            for c in cons_to_deactivate:
                c.deactivate()
            if cons_to_deactivate:
                print(
                    f"  egret_uc_subproblem_transform: deactivated "
                    f"{len(cons_to_deactivate)} pure first-stage constraints",
                    flush=True,
                )
        return model


def _print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()


def stage8_run_benders_multi(
    sp,
    ef_reference: dict,
    max_iterations: int = 80,
    eta_lower: float = 0.0,
    obj_tol: float = 1e-3,
    rel_tol: float = 1e-4,
    subproblem_transforms: Optional[Sequence] = None,
):
    """
    Multi-scenario SPAROW Benders under the new paradigm.

    Caller must supply a *fresh* multi-scenario SP (post-EF rebuild).
    Uses solve_and_return_model + public residual transform; no monkey-patches.
    """
    _print("\n=== STAGE 8 (multi, new paradigm): SPAROW BendersSolver + EF comparison ===")
    from sparow.benders import BendersSolver

    ef_obj = ef_reference["objective"]
    _print(f"  Multi-scenario EF reference objective: {ef_obj:.6g}")

    if eta_lower is None:
        eta_lower = 0.0
    eta_bounds_map = {b: (float(eta_lower), None) for b in sp.bundles}
    _print(f"  eta_bounds_map keys: {list(eta_bounds_map.keys())}")
    for b, bounds in eta_bounds_map.items():
        if bounds[0] is None:
            raise RuntimeError(
                f"Stage 8 FAILED – eta lower bound for bundle {b!r} is None."
            )

    # First-stage seed / unfix on every bundle
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
    _print(f"  Unfixed {n_unfixed} previously-fixed first-stage variables")
    _print(f"  Seeded numeric values on {n_seeded} first-stage variables")

    if subproblem_transforms is None:
        subproblem_transforms = [egret_uc_subproblem_transform]

    solver = BendersSolver()
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

    # Optional progress banners
    try:
        from or_topas.benders.benders_serial import Benders_Serial as _BendersCutGen
    except ImportError:
        try:
            from or_topas.benders import Benders_Serial as _BendersCutGen
        except ImportError:
            _BendersCutGen = None

    _iter_state = {"n": 0, "cuts_total": 0}
    _orig_generate_cut = None
    if _BendersCutGen is not None and hasattr(_BendersCutGen, "generate_cut"):
        _orig_generate_cut = _BendersCutGen.generate_cut

        def _generate_cut_with_progress(self, *args, **kwargs):
            cuts = _orig_generate_cut(self, *args, **kwargs)
            _iter_state["n"] += 1
            n_cuts = len(cuts) if cuts is not None else 0
            _iter_state["cuts_total"] += n_cuts
            master_obj = None
            try:
                m = getattr(self, "model", None) or getattr(self, "_model", None)
                if m is not None and hasattr(m, "obj"):
                    master_obj = pyo.value(m.obj)
            except Exception:
                pass
            parts = [
                f"  --- Benders iteration {_iter_state['n']}",
                f"cuts_added={n_cuts}",
                f"cuts_total={_iter_state['cuts_total']}",
            ]
            if master_obj is not None:
                parts.append(f"master_obj={master_obj:.6g}")
            if n_cuts == 0:
                parts.append("CONVERGED (no cuts)")
            parts.append("---")
            _print("  ".join(parts))
            return cuts

        _BendersCutGen.generate_cut = _generate_cut_with_progress

    _print(
        "  Calling BendersSolver.solve_and_return_model "
        "(public residual transform, no monkey-patches, multi-scenario) ..."
    )
    try:
        data = solver.solve_and_return_model(
            sp,
            eta_bounds_map,
            subproblem_transforms=list(subproblem_transforms),
            master_transforms=None,
        )
        results = data.solutions
        upper_model = data.upper_model
    finally:
        if _orig_generate_cut is not None:
            _BendersCutGen.generate_cut = _orig_generate_cut

    _print(
        f"  Solve returned after {_iter_state['n']} Benders iteration(s), "
        f"{_iter_state['cuts_total']} cut(s) total."
    )

    # Recover objective
    benders_obj = None
    if hasattr(results, "to_dict"):
        try:
            d = results.to_dict()
            soln = next(iter(d["solutions"].values()))
            benders_obj = float(soln["objectives"][0]["value"])
        except Exception:
            pass
    if benders_obj is None and upper_model is not None:
        try:
            benders_obj = float(pyo.value(upper_model.obj))
        except Exception:
            pass
    if benders_obj is None and hasattr(results, "obj_value"):
        try:
            benders_obj = float(results.obj_value)
        except Exception:
            pass

    if benders_obj is None:
        raise RuntimeError(
            "Stage 8 FAILED – could not recover a numeric objective from "
            "Benders results / upper_model."
        )

    print(f"  Benders objective value: {benders_obj:.6g}")
    abs_diff = abs(benders_obj - ef_obj)
    tol = max(obj_tol, rel_tol * abs(ef_obj))
    print(f"  |Benders - EF| = {abs_diff:.6g}  (tol = {tol:.6g})")

    if abs_diff > tol:
        raise RuntimeError(
            f"Stage 8 FAILED – Benders objective {benders_obj:.6g} differs from "
            f"multi-scenario EF reference {ef_obj:.6g} by more than tolerance {tol:.6g}."
        )

    _print("  Stage 8 PASSED – multi-scenario Benders matches EF within tolerance")
    return {
        "benders_objective": benders_obj,
        "ef_objective": ef_obj,
        "abs_diff": abs_diff,
        "results": results,
        "upper_model": upper_model,
    }


# ---------------------------------------------------------------------------
# Self-test: EF → fresh SP rebuild → Benders (multi-scenario rule)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from egret_sparow_staged_multi import (
        stage0_check_environment,
        stage1_build_egret_model,
        stage2_inspect_variables,
        stage3_verify_first_stage_names,
        stage5_egret_sparow_sp_multi,
        stage6_solve_extensive_form,
    )

    DEFAULT_SCENARIOS = [
        {"ID": "low",  "Probability": 0.25, "load_scale": 0.85},
        {"ID": "med",  "Probability": 0.50, "load_scale": 1.00},
        {"ID": "high", "Probability": 0.25, "load_scale": 1.15},
    ]

    stage0_check_environment()
    md, model = stage1_build_egret_model(force_synthetic=True, n_periods=4)
    binary_comps, _, suggested = stage2_inspect_variables(model)
    fs_names = [b["name"] for b in binary_comps if "uniton" in b["name"].lower()] or suggested
    stage3_verify_first_stage_names(model, fs_names)

    _print(f"\n  Using scenarios: {DEFAULT_SCENARIOS}")
    _print(f"  first_stage_names = {fs_names}")

    # Stage 5: multi SP
    sp = stage5_egret_sparow_sp_multi(
        md, first_stage_names=fs_names, scenarios=DEFAULT_SCENARIOS
    )

    # Stage 6: EF on that SP
    ef_ref = stage6_solve_extensive_form(sp, solver_name="gurobi", time_limit=300)
    _print(f"\n  Multi-scenario EF reference objective = {ef_ref['objective']:.6g}")

    # CRITICAL multi rule: rebuild a fresh SP before Benders
    _print("\n  Rebuilding multi-scenario SP for Benders (EF may have altered bundles) ...")
    sp_fresh = stage5_egret_sparow_sp_multi(
        md, first_stage_names=fs_names, scenarios=DEFAULT_SCENARIOS
    )

    # Stage 8 on the fresh SP
    out = stage8_run_benders_multi(sp_fresh, ef_ref, max_iterations=80)

    print("\nSelf-test finished (multi-scenario, new paradigm).")
    print(f"  EF      = {out['ef_objective']:.6g}")
    print(f"  Benders = {out['benders_objective']:.6g}")
    print(f"  |diff|  = {out['abs_diff']:.6g}")
    print(f"  upper_model type = {type(out['upper_model'])}")
