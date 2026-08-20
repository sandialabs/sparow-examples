#!/usr/bin/env python3
"""
stage8_new_paradigm.py
======================
Drop-in replacement for Stage 8 of egret_sparow_staged.py under the new
paradigm (no class-method monkey-patches).

Contains:
  - egret_uc_subproblem_transform  (thin public residual transform)
  - stage8_run_benders             (uses solve_and_return_model + public transform)

Usage after Phase-0 inspection
------------------------------
1. Review / optionally adjust egret_uc_subproblem_transform once you know
   whether official relax_second_stage is sufficient.
2. Replace the existing stage8_run_benders in egret_sparow_staged.py with
   the version below (or import from this module).
3. Re-run:

   python -c '
   from egret_sparow_staged import run_staged_diagnostics
   run_staged_diagnostics(
       force_synthetic=True, n_periods=4, first_stage_names=["UnitOn"],
       stop_after_ef=False, run_benders=True, max_benders_iterations=50,
   )
   '

Expected: EF objective ≈ 5886.4 and Benders matches within tolerance.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence

import pyomo.environ as pyo


def _print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Thin public residual transform
# (replaces the old BendersSolver._setup_topas_subproblem monkey-patch)
# ---------------------------------------------------------------------------
def egret_uc_subproblem_transform(sp, model):
    """
    Public subproblem transform for EGRET UC under classical Benders.

    - Residual Binary / Integer → UnitInterval (required for dual cuts)
    - Deactivate pure first-stage-only constraints (they belong only on the master)

    Pass via:
        solver.solve_and_return_model(..., subproblem_transforms=[egret_uc_subproblem_transform])

    This is deliberately a conservative superset of the old monkey-patch so it
    remains correct even if the official sparow.sp.util.relax_second_stage is
    incomplete.  After Phase-0 inspection we can simplify if the official util
    already covers both responsibilities.
    """
    from pyomo.common.collections import ComponentSet
    from pyomo.core.expr.visitor import identify_variables

    # 1. Residual Binary / Integer → continuous [0, 1]
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
        _print(
            f"  egret_uc_subproblem_transform: relaxed {n_relaxed} residual "
            "Binary/Integer vars → UnitInterval"
        )

    # 2. Deactivate pure first-stage-only constraints
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
            _print(
                f"  egret_uc_subproblem_transform: deactivated "
                f"{len(cons_to_deactivate)} pure first-stage constraints"
            )

    return model


# ---------------------------------------------------------------------------
# Replacement Stage 8
# ---------------------------------------------------------------------------
def stage8_run_benders(
    sp,
    ef_reference: dict,
    max_iterations: int = 50,
    eta_lower: float = 0.0,
    obj_tol: float = 1e-3,
    rel_tol: float = 1e-4,
    subproblem_transforms: Optional[Sequence] = None,
):
    """
    SPAROW Benders under the new paradigm:

      - solve_and_return_model (official upper_model; no capture monkey-patch)
      - public residual transform via subproblem_transforms
      - no monkey-patches of _setup_topas_subproblem or add_subproblem

    Returns a dict that includes the live upper_model ready for AOS-Benders.
    """
    _print("\n=== STAGE 8: SPAROW BendersSolver (new paradigm + EF comparison) ===")
    from sparow.benders import BendersSolver

    ef_obj = ef_reference["objective"]
    _print(f"  EF reference objective: {ef_obj:.6g}")

    if eta_lower is None:
        eta_lower = 0.0
    eta_bounds_map = {b: (float(eta_lower), None) for b in sp.bundles}
    _print(f"  eta_bounds_map keys: {list(eta_bounds_map.keys())}")
    for b, bounds in eta_bounds_map.items():
        if bounds[0] is None:
            raise RuntimeError(
                f"Stage 8 FAILED – eta lower bound for bundle {b!r} is None."
            )

    # ------------------------------------------------------------------
    # First-stage seed / unfix (still required for standard_lp)
    # ------------------------------------------------------------------
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

    # Default residual transform if the caller did not supply one
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

    # Optional progress banner (non-critical)
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
        "(public residual transform, no monkey-patches) ..."
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
            f"EF reference {ef_obj:.6g} by more than tolerance {tol:.6g}."
        )

    _print("  Stage 8 PASSED – Benders matches EF within tolerance")
    return {
        "benders_objective": benders_obj,
        "ef_objective": ef_obj,
        "abs_diff": abs_diff,
        "results": results,
        "upper_model": upper_model,  # ready for AOS-Benders
    }


# ---------------------------------------------------------------------------
# Convenience: run only Stage 8 on an already-built SP + EF reference
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Self-test for SINGLE-scenario path.
    #
    # CRITICAL ordering (from the original EGRET+SPAROW work and Phase-0):
    #   1. Build SP
    #   2. Solve EF on that SP first
    #   3. Run Benders on the *same* SP
    #
    # copy.deepcopy on a *fresh* EGRET SP still hard-crashes
    # (cannot pickle dict_keys / _parent). After EF the same deepcopy only
    # warns. Multi-scenario is the opposite rule (rebuild a fresh SP after EF).
    # ------------------------------------------------------------------
    from egret_sparow_staged import (
        stage0_check_environment,
        stage1_build_egret_model,
        stage2_inspect_variables,
        stage3_verify_first_stage_names,
        stage5_egret_sparow_sp,
        stage6_solve_extensive_form,
    )

    stage0_check_environment()
    md, model = stage1_build_egret_model(force_synthetic=True, n_periods=4)
    binary_comps, _, suggested = stage2_inspect_variables(model)
    fs_names = [b["name"] for b in binary_comps if "uniton" in b["name"].lower()] or suggested
    stage3_verify_first_stage_names(model, fs_names)
    sp = stage5_egret_sparow_sp(
        md, first_stage_names=fs_names, scenario_scales={"base": 1.0}
    )

    # Step 2 – EF first (softens Contingencies / uncopyable fields)
    ef_ref = stage6_solve_extensive_form(sp, solver_name="gurobi", time_limit=120)
    print(f"\n  EF reference objective = {ef_ref['objective']:.6g}")

    # Step 3 – Benders on the same SP
    out = stage8_run_benders(sp, ef_ref, max_iterations=50)

    print("\nSelf-test finished.")
    print(f"  EF      = {out['ef_objective']:.6g}")
    print(f"  Benders = {out['benders_objective']:.6g}")
    print(f"  |diff|  = {out['abs_diff']:.6g}")
    print(f"  upper_model type = {type(out['upper_model'])}")
