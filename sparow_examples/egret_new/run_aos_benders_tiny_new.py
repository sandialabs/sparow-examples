#!/usr/bin/env python3
"""
run_aos_benders_tiny_new.py
===========================
Sandboxed Path-B (AOS-Benders) for the synthetic tiny UC under the new paradigm.

- Builds the SP the same way as the original PoC
- Obtains the live upper_model via the cleaned Stage-8 (solve_and_return_model
  + public residual transform; no capture / _setup monkey-patches)
- Feeds that upper_model straight into aos_benders_generate_candidates
  (enumeration_method="gurobi_pool") + aos_benders_filter

All new code lives under artifacts/aos_update/.  Originals under attachments/
are never modified.

Usage
-----
  # From a directory that has the original staged helpers + aos_benders on PYTHONPATH
  PYTHONPATH=/path/to/attachments:/path/to/artifacts/aos_update:$PYTHONPATH \
    python run_aos_benders_tiny_new.py

  # Or with explicit flags
  python run_aos_benders_tiny_new.py --n-periods 4 --num-sols 12 --rel-gap 0.0
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyomo.environ as pyo

# ---------------------------------------------------------------------------
# Import path: original attachments first, then this sandbox
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ATTACHMENTS = _HERE.parent.parent / "attachments"   # /home/workdir/attachments
_CANDIDATE_PATHS = [
    str(_ATTACHMENTS),
    str(_HERE),
    str(_ATTACHMENTS / "aos_analysis"),
    "/home/workdir/attachments",
    "/home/workdir/artifacts/aos_analysis",
]
for p in _CANDIDATE_PATHS:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("aos_tiny_new")


def _print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Pool pretty-printing (identical spirit to the original PoC)
# ---------------------------------------------------------------------------

def _sol_objective(sol) -> Optional[float]:
    try:
        if hasattr(sol, "objective") and callable(sol.objective):
            obj = sol.objective()
            return float(obj.value if hasattr(obj, "value") else obj)
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
            for v in sol_or_model.component_data_objects(pyo.Var, active=True, descend_into=True):
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


# ---------------------------------------------------------------------------
# Shared build helper (same contract as original)
# ---------------------------------------------------------------------------

def _build_tiny_uc_sp(n_periods: int = 4, first_stage_names: Optional[Sequence[str]] = None):
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
        md, first_stage_names=first_stage_names, scenario_scales={"base": 1.0}
    )
    return sp, list(first_stage_names)


# ---------------------------------------------------------------------------
# Path B under the new paradigm
# ---------------------------------------------------------------------------

def run_aos_benders_egret_new(
    n_periods: int = 4,
    first_stage_names: Optional[Sequence[str]] = None,
    num_solutions: int = 12,
    rel_opt_gap: float = 0.0,
    max_benders_iterations: int = 50,
    tee: bool = False,
) -> Dict[str, Any]:
    """
    Full Path B (new paradigm):

      1. Build synthetic SP + EF reference
      2. Run cleaned Stage-8 (solve_and_return_model + public residual transform)
      3. Hand the returned upper_model to aos_benders_generate_candidates
         (gurobi_pool) + aos_benders_filter
    """
    _print("\n" + "=" * 72)
    _print("PATH B (new paradigm): AOS-Benders on synthetic tiny UC")
    _print("=" * 72)

    from egret_sparow_staged import stage6_solve_extensive_form
    # Prefer the sandboxed Stage-8; fall back to a local import if needed
    try:
        from stage8_new_paradigm import stage8_run_benders, egret_uc_subproblem_transform
    except ImportError:
        # allow running when this file is the only entry point
        sys.path.insert(0, str(_HERE))
        from stage8_new_paradigm import stage8_run_benders, egret_uc_subproblem_transform

    sp, fs_names = _build_tiny_uc_sp(n_periods, first_stage_names)
    ef_ref = stage6_solve_extensive_form(sp, solver_name="gurobi", time_limit=180)
    _print(f"  EF reference objective = {ef_ref['objective']:.6g}")

    # --- cleaned Stage-8 (no monkey-patches) ---
    try:
        benders_out = stage8_run_benders(
            sp,
            ef_reference=ef_ref,
            max_iterations=max_benders_iterations,
            subproblem_transforms=[egret_uc_subproblem_transform],
        )
        upper_model = benders_out["upper_model"]
    except Exception as e:
        _print(f"\n  *** Stage-8 / Benders failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "sp": sp, "ef_reference": ef_ref}

    # --- AOS-Benders (gurobi_pool kernel) ---
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
        "benders_out": benders_out,
    }


# ---------------------------------------------------------------------------
# Optional: still keep the Farmer kernel as a non-EGRET sanity check
# ---------------------------------------------------------------------------

def run_aos_benders_kernel_demo(
    num_solutions: int = 10,
    rel_opt_gap: float = 0.01,
    tee: bool = False,
):
    _print("\n" + "=" * 72)
    _print("KERNEL DEMO: AOS-Benders with gurobi_pool on Farmer (unchanged)")
    _print("=" * 72)
    try:
        from aos_benders import aos_benders_generate_candidates, aos_benders_filter
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
        description="AOS-Benders (new paradigm) on synthetic tiny UC"
    )
    parser.add_argument("--n-periods", type=int, default=4)
    parser.add_argument("--num-sols", type=int, default=12)
    parser.add_argument("--rel-gap", type=float, default=0.0)
    parser.add_argument("--max-benders-iter", type=int, default=50)
    parser.add_argument("--skip-egret-benders", action="store_true")
    parser.add_argument("--skip-farmer-kernel", action="store_true")
    parser.add_argument("--tee", action="store_true", default=False)
    args = parser.parse_args()

    results = {}
    if not args.skip_egret_benders:
        results["egret_aos_benders"] = run_aos_benders_egret_new(
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
    _print("*** Finished (new-paradigm Path B). Inspect the pools above. ***")
    _print("=" * 72)
    return results


if __name__ == "__main__":
    main()
