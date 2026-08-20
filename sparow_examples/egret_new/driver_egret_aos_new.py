#!/usr/bin/env python3
"""
driver_egret_aos_new.py
=======================
Single-scenario EGRET AOS driver under the **new paradigm**.

Changes vs the original driver_egret_aos.py
-------------------------------------------
- Residual handling: locked public `egret_uc_subproblem_transform` via
  `subproblem_transforms=` (see TRANSFORM_DECISION_RECORD.md).
- Master capture: `solve_and_return_model` returns `upper_model` directly.
  No monkey-patches of `_setup_topas_subproblem` or `add_subproblem`.
- Single-scenario operational rule: EF first on the SP, then Benders on the
  *same* SP (softens Contingencies deepcopy).

Everything else (CLI matrix, CSV schema, AOS extensive / AOS-Benders) is
preserved so results remain comparable to the original driver.

Usage (same directory as egret_sparow_staged.py + stage8_new_paradigm.py):

  python driver_egret_aos_new.py --out_dir results_new/ --intensity 1 \\
      --force-synthetic --n-periods 4 2>&1 | tee driver_single_new.log
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyomo.environ as pyo

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (_SCRIPT_DIR, os.getcwd()):
    if p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

INTENSITY_GAPS = {
    1: [0.0],
    2: [0.0, 0.01, 0.05],
    3: [0.0, 0.01, 0.05, 0.10],
}

AOS_SOLVER = "gurobi"
CSV_FIELDS = [
    "case", "n_periods", "first_stage_mode", "solve_mode", "aos_method",
    "rel_gap", "num_solutions_cap", "time_limit_s", "base_time_s", "aos_time_s",
    "total_time_s", "n_candidates", "n_true", "obj_value", "master_lb_or_eta",
    "n_cuts", "n_iterations", "n_first_stage", "status", "error_msg",
    "solutions_summary",
]


def _print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Helpers (solution summary / UnitOn compact)
# ---------------------------------------------------------------------------
def _sol_objective(sol) -> Optional[float]:
    """Robust objective extraction for or_topas / PyomoPoolManager solution objects."""
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
        if isinstance(sol, dict) and "objective" in sol:
            return float(sol["objective"])
        if hasattr(sol, "obj"):
            return float(pyo.value(sol.obj))
    except Exception:
        pass
    return None


def _extract_uniton_compact(sol_or_model) -> str:
    """Return a compact UnitOn bit-string from pool solution or Pyomo model."""
    vals = []
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
                    vals.append(int(round(float(val))) if val is not None else "?")
                except Exception:
                    vals.append("?")
            if vals:
                return "(" + ", ".join(str(v) for v in vals) + ")"

        if hasattr(sol_or_model, "get_values"):
            d = sol_or_model.get_values()
            for k in sorted(d.keys(), key=str):
                if "uniton" in str(k).lower():
                    vals.append(int(round(float(d[k]))))
            if vals:
                return "(" + ", ".join(str(v) for v in vals) + ")"

        for v in sol_or_model.component_data_objects(pyo.Var, active=True, descend_into=True):
            if "uniton" in v.name.lower():
                try:
                    vals.append(int(round(float(pyo.value(v)))))
                except Exception:
                    vals.append("?")
        if vals:
            return "(" + ", ".join(str(v) for v in vals) + ")"
    except Exception:
        pass
    return "()"


def _format_solution_summary(pool, threshold: int) -> str:
    if pool is None:
        return ""
    try:
        n = len(pool)
    except Exception:
        return str(pool)[:80]
    if n == 0:
        return "[]"
    if n > threshold:
        return f"[{n} sols; omitted (threshold={threshold})]"
    parts = []
    for i, sol in enumerate(pool):
        obj = _sol_objective(sol)
        bits = _extract_uniton_compact(sol)
        parts.append(f"{i}:obj={obj} UnitOn={bits}")
    return "[" + "; ".join(parts) + "]"


def _case_label(egret_file, n_periods, force_synthetic) -> str:
    if force_synthetic or egret_file is None:
        return f"synthetic_T{n_periods if n_periods else 4}"
    return f"egret:{Path(egret_file).stem}_T{n_periods if n_periods else 'full'}"


# ---------------------------------------------------------------------------
# SP build (delegates to staged helpers)
# ---------------------------------------------------------------------------
def build_egret_sp(
    n_periods: Optional[int] = 4,
    include_start_stop: bool = False,
    egret_file: Optional[str] = None,
    force_synthetic: bool = False,
    first_stage_names: Optional[Sequence[str]] = None,
) -> Tuple[Any, List[str]]:
    from egret_sparow_staged import (
        stage0_check_environment,
        stage1_build_egret_model,
        stage2_inspect_variables,
        stage3_verify_first_stage_names,
        stage5_egret_sparow_sp,
    )

    stage0_check_environment()
    use_synthetic = bool(force_synthetic) or (egret_file is None)

    if not use_synthetic and egret_file is not None and n_periods is not None:
        _print(
            f"  WARNING: truncating horizon of EGRET JSON '{egret_file}' "
            f"to first {n_periods} periods"
        )

    if use_synthetic:
        md, model = stage1_build_egret_model(
            force_synthetic=True,
            n_periods=n_periods if n_periods is not None else 4,
        )
    else:
        md, model = stage1_build_egret_model(
            explicit_path=egret_file,
            force_synthetic=False,
            allow_synthetic=False,
            allow_github_fetch=False,
            n_periods=n_periods,
        )

    binary_comps, _, suggested = stage2_inspect_variables(model)
    if first_stage_names is None:
        names = [b["name"] for b in binary_comps]
        if include_start_stop:
            first_stage_names = [
                n for n in names
                if any(k in n.lower() for k in ("uniton", "unitstart", "unitstop"))
            ] or suggested
        else:
            first_stage_names = [n for n in names if "uniton" in n.lower()] or suggested
    else:
        first_stage_names = list(first_stage_names)

    stage3_verify_first_stage_names(model, first_stage_names)
    _print(f"  first_stage_names = {list(first_stage_names)}")

    sp = stage5_egret_sparow_sp(
        md, first_stage_names=first_stage_names, scenario_scales={"base": 1.0}
    )
    return sp, list(first_stage_names)


# ---------------------------------------------------------------------------
# Baseline solves (new paradigm for Benders)
# ---------------------------------------------------------------------------
def run_extensive_base(sp, time_limit: float) -> Dict[str, Any]:
    """
    Solve the extensive form via sparow.ef.ExtensiveFormSolver.
    Returns the live EF *model* so EF AOS (gurobi_pool / binary) can run on it,
    plus an ef_reference dict for the single-scenario Benders rule.
    """
    import math
    from sparow.ef import ExtensiveFormSolver

    t0 = time.perf_counter()
    ef_solver = ExtensiveFormSolver()
    ef_solver.set_options(
        solver="gurobi",
        solver_options={"TimeLimit": time_limit, "timelimit": time_limit},
    )
    status = "optimal"
    obj = float("nan")
    model = None
    try:
        res = ef_solver.solve_and_return_EF(sp)
        model = getattr(res, "model", None)
        # Prefer objective from results, fall back to model
        try:
            rd = res.to_dict() if hasattr(res, "to_dict") else None
            if rd and "solutions" in rd and rd["solutions"]:
                soln = next(iter(rd["solutions"].values()))
                if isinstance(soln, dict) and soln.get("objectives"):
                    obj = float(soln["objectives"][0]["value"])
        except Exception:
            pass
        if (obj != obj) and model is not None:  # NaN check
            objs = list(
                model.component_data_objects(
                    pyo.Objective, active=True, descend_into=False
                )
            )
            if objs:
                obj = float(pyo.value(objs[0]))
        if model is not None and (obj != obj):
            try:
                obj = float(pyo.value(model.obj))
            except Exception:
                pass
    except Exception as exc:
        return {
            "model": None,
            "ef_reference": None,
            "base_time_s": time.perf_counter() - t0,
            "obj_value": float("nan"),
            "master_lb_or_eta": float("nan"),
            "n_cuts": float("nan"),
            "n_iterations": float("nan"),
            "status": "error",
            "error_msg": f"{type(exc).__name__}: {str(exc)[:200]}",
        }

    elapsed = time.perf_counter() - t0
    if elapsed > time_limit + 1.0:
        status = "timeout"

    ef_reference = {"objective": obj, "model": model}
    return {
        "model": model,  # live EF model for EF AOS
        "ef_reference": ef_reference,
        "base_time_s": elapsed,
        "obj_value": obj,
        "master_lb_or_eta": obj,
        "n_cuts": 0.0,
        "n_iterations": 1.0,
        "status": status,
        "error_msg": "",
    }


def run_benders_base(
    sp,
    time_limit: float,
    max_iterations: int = 80,
    eta_lower: float = 0.0,
    ef_reference: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    New-paradigm Benders base:
      - public residual transform via stage8_run_benders
      - solve_and_return_model → live upper_model (no capture monkey-patch)
      - single-scenario rule: EF must already have been run on this SP
        (caller supplies ef_reference, or we run EF first here)
    """
    from stage8_new_paradigm import stage8_run_benders
    from egret_sparow_staged import stage6_solve_extensive_form

    t0 = time.perf_counter()
    try:
        if ef_reference is None:
            _print("  [new paradigm] EF first on same SP (single-scenario rule) ...")
            ef_reference = stage6_solve_extensive_form(
                sp, solver_name="gurobi", time_limit=min(time_limit, 180)
            )

        out = stage8_run_benders(
            sp,
            ef_reference,
            max_iterations=max_iterations,
            eta_lower=eta_lower,
        )
        elapsed = time.perf_counter() - t0
        status = "timeout" if elapsed > time_limit + 1.0 else "optimal"
        return {
            "model": out["upper_model"],
            "ef_reference": ef_reference,
            "base_time_s": elapsed,
            "obj_value": float(out["benders_objective"]),
            "master_lb_or_eta": float(out["benders_objective"]),
            "n_cuts": float("nan"),
            "n_iterations": float("nan"),
            "status": status,
            "error_msg": "",
        }
    except Exception as exc:
        return {
            "model": None,
            "ef_reference": ef_reference,
            "base_time_s": time.perf_counter() - t0,
            "obj_value": float("nan"),
            "master_lb_or_eta": float("nan"),
            "n_cuts": float("nan"),
            "n_iterations": float("nan"),
            "status": "error",
            "error_msg": f"{type(exc).__name__}: {str(exc)[:240]}",
        }


# ---------------------------------------------------------------------------
# AOS runners
# ---------------------------------------------------------------------------
def run_aos_extensive(
    model,
    aos_method: str,
    rel_gap: float,
    num_solutions: int,
    time_limit: float,
    keep_pool: bool = True,
) -> Dict[str, Any]:
    """
    EF AOS on the live extensive-form model returned by run_extensive_base.
    Uses or_topas.aos.gurobi_generate_solutions (PoolSearchMode=2) or binary
    enumeration — same path as the original driver.
    """
    t0 = time.perf_counter()
    if model is None:
        return {
            "aos_time_s": 0.0,
            "n_candidates": float("nan"),
            "n_true": float("nan"),
            "obj_value": float("nan"),
            "status": "error",
            "error_msg": "EF AOS requires a live EF model from run_extensive_base",
            "pool": None,
        }

    status = "optimal"
    pool = None
    try:
        if aos_method == "gurobi_pool":
            try:
                from or_topas.aos import gurobi_generate_solutions
            except ImportError:
                from gurobi_solnpool import gurobi_generate_solutions
            pool = gurobi_generate_solutions(
                model=model,
                num_solutions=num_solutions,
                rel_opt_gap=rel_gap,
                pool_search_mode=2,
                tee=False,
            )
        elif aos_method == "binary":
            try:
                from or_topas.aos import enumerate_binary_solutions
            except ImportError:
                from balas import enumerate_binary_solutions
            pool = enumerate_binary_solutions(
                model,
                num_solutions=num_solutions,
                rel_opt_gap=rel_gap,
                solver=AOS_SOLVER,
                tee=False,
            )
        else:
            raise ValueError(f"Unknown aos_method for extensive: {aos_method}")
    except Exception as exc:
        return {
            "aos_time_s": time.perf_counter() - t0,
            "n_candidates": float("nan"),
            "n_true": float("nan"),
            "obj_value": float("nan"),
            "status": "error",
            "error_msg": f"{type(exc).__name__}: {str(exc)[:200]}",
            "pool": None,
        }

    elapsed = time.perf_counter() - t0
    if elapsed > time_limit:
        status = "timeout"

    n = len(pool) if pool is not None else 0
    obj0 = _sol_objective(pool[0]) if pool else float("nan")
    return {
        "aos_time_s": elapsed,
        "n_candidates": n,
        "n_true": n,  # EF pool is already the true set
        "obj_value": obj0 if obj0 is not None else float("nan"),
        "status": status,
        "error_msg": "",
        "pool": pool if keep_pool else None,
    }


def run_aos_benders(
    upper_model,
    aos_method: str,
    rel_gap: float,
    num_solutions: int,
    time_limit: float,
    keep_pool: bool = True,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    enumeration_method = "gurobi_pool" if aos_method == "gurobi_pool" else "binary"
    try:
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

        candidate_pool, data = aos_benders_generate_candidates(
            m=upper_model,
            rel_gap=rel_gap,
            num_solutions=num_solutions,
            mip_solver=AOS_SOLVER,
            enumeration_method=enumeration_method,
            tee=False,
        )
        if time.perf_counter() - t0 > time_limit:
            return {
                "aos_time_s": time.perf_counter() - t0,
                "n_candidates": len(candidate_pool),
                "n_true": float("nan"),
                "obj_value": float("nan"),
                "status": "timeout",
                "error_msg": "",
                "pool": None,
            }
        true_pool = aos_benders_filter(candidate_pool, data, tee=False, tee_final=False)
        elapsed = time.perf_counter() - t0
        lb = float("nan")
        try:
            lb = float(data.lower_bound)
        except Exception:
            pass
        return {
            "aos_time_s": elapsed,
            "n_candidates": len(candidate_pool),
            "n_true": len(true_pool),
            "obj_value": lb,
            "status": "timeout" if elapsed > time_limit else "optimal",
            "error_msg": "",
            "pool": true_pool if keep_pool else None,
        }
    except Exception as exc:
        return {
            "aos_time_s": time.perf_counter() - t0,
            "n_candidates": float("nan"),
            "n_true": float("nan"),
            "obj_value": float("nan"),
            "status": "error",
            "error_msg": f"{type(exc).__name__}: {str(exc)[:200]}",
            "pool": None,
        }


# ---------------------------------------------------------------------------
# One configuration
# ---------------------------------------------------------------------------
def run_one_config(
    *,
    n_periods: Optional[int],
    include_start_stop: bool,
    solve_mode: str,
    aos_method: str,
    rel_gap: float,
    num_solutions: int,
    time_limit: float,
    max_benders_iterations: int,
    summary_threshold: int,
    case: str,
    egret_file: Optional[str] = None,
    force_synthetic: bool = False,
    first_stage_names: Optional[Sequence[str]] = None,
    cached_ef_base: Optional[Dict[str, Any]] = None,
    cached_benders_base: Optional[Dict[str, Any]] = None,
    sp: Any = None,
) -> Dict[str, Any]:
    fs_label = "UnitOn+StartStop" if include_start_stop else "UnitOn"
    row: Dict[str, Any] = {
        "case": case,
        "n_periods": n_periods if n_periods is not None else float("nan"),
        "first_stage_mode": fs_label,
        "solve_mode": solve_mode,
        "aos_method": aos_method,
        "rel_gap": rel_gap,
        "num_solutions_cap": num_solutions,
        "time_limit_s": time_limit,
        "base_time_s": float("nan"),
        "aos_time_s": float("nan"),
        "total_time_s": float("nan"),
        "n_candidates": float("nan"),
        "n_true": float("nan"),
        "obj_value": float("nan"),
        "master_lb_or_eta": float("nan"),
        "n_cuts": float("nan"),
        "n_iterations": float("nan"),
        "n_first_stage": float("nan"),
        "status": "error",
        "error_msg": "",
        "solutions_summary": "",
    }

    try:
        if solve_mode == "extensive":
            base = cached_ef_base
            if base is None:
                if sp is None:
                    sp, _ = build_egret_sp(
                        n_periods=n_periods,
                        include_start_stop=include_start_stop,
                        egret_file=egret_file,
                        force_synthetic=force_synthetic,
                        first_stage_names=first_stage_names,
                    )
                base = run_extensive_base(sp, time_limit=time_limit)
            row["n_first_stage"] = (
                len(getattr(sp, "int_to_FirstStageVar", {}).get(next(iter(sp.bundles)), {}))
                if sp is not None else float("nan")
            )
            row["base_time_s"] = base["base_time_s"]
            row["obj_value"] = base["obj_value"]
            row["master_lb_or_eta"] = base["master_lb_or_eta"]
            row["status"] = base["status"]
            row["error_msg"] = base.get("error_msg", "")

            if base["status"] == "optimal" and aos_method != "baseline":
                aos = run_aos_extensive(
                    base.get("model"), aos_method, rel_gap, num_solutions, time_limit
                )
                row["aos_time_s"] = aos["aos_time_s"]
                row["n_candidates"] = aos["n_candidates"]
                row["n_true"] = aos["n_true"]
                if aos.get("obj_value") == aos.get("obj_value"):  # not NaN
                    row["obj_value"] = aos["obj_value"]
                row["status"] = aos["status"]
                row["error_msg"] = aos.get("error_msg", "")
                row["solutions_summary"] = _format_solution_summary(
                    aos.get("pool"), summary_threshold
                )
            elif aos_method == "baseline":
                row["n_candidates"] = 1
                row["n_true"] = 1
                row["aos_time_s"] = 0.0
                row["solutions_summary"] = "baseline"

        else:  # benders
            base = cached_benders_base
            if base is None:
                if sp is None:
                    sp, _ = build_egret_sp(
                        n_periods=n_periods,
                        include_start_stop=include_start_stop,
                        egret_file=egret_file,
                        force_synthetic=force_synthetic,
                        first_stage_names=first_stage_names,
                    )
                # Single-scenario rule: EF first, then Benders on same SP
                ef_base = cached_ef_base
                if ef_base is None or ef_base.get("ef_reference") is None:
                    ef_base = run_extensive_base(sp, time_limit=time_limit)
                base = run_benders_base(
                    sp,
                    time_limit=time_limit,
                    max_iterations=max_benders_iterations,
                    ef_reference=ef_base.get("ef_reference"),
                )
            row["n_first_stage"] = (
                len(getattr(sp, "int_to_FirstStageVar", {}).get(next(iter(sp.bundles)), {}))
                if sp is not None else float("nan")
            )
            row["base_time_s"] = base["base_time_s"]
            row["obj_value"] = base["obj_value"]
            row["master_lb_or_eta"] = base["master_lb_or_eta"]
            row["n_iterations"] = base.get("n_iterations", float("nan"))
            row["status"] = base["status"]
            row["error_msg"] = base.get("error_msg", "")

            if base["status"] == "optimal" and aos_method != "baseline":
                upper = base.get("model")
                if upper is None:
                    row["status"] = "error"
                    row["error_msg"] = "no upper_model from Benders base"
                else:
                    aos = run_aos_benders(
                        upper, aos_method, rel_gap, num_solutions, time_limit
                    )
                    row["aos_time_s"] = aos["aos_time_s"]
                    row["n_candidates"] = aos["n_candidates"]
                    row["n_true"] = aos["n_true"]
                    if aos.get("obj_value") == aos.get("obj_value"):
                        row["obj_value"] = aos["obj_value"]
                    row["status"] = aos["status"]
                    row["error_msg"] = aos.get("error_msg", "")
                    row["solutions_summary"] = _format_solution_summary(
                        aos.get("pool"), summary_threshold
                    )
            elif aos_method == "baseline":
                row["n_candidates"] = 1
                row["n_true"] = 1
                row["aos_time_s"] = 0.0
                row["solutions_summary"] = "baseline"

        row["total_time_s"] = float(row["base_time_s"] or 0) + float(
            row["aos_time_s"] if row["aos_time_s"] == row["aos_time_s"] else 0
        )
    except Exception as exc:
        row["status"] = "error"
        row["error_msg"] = f"{type(exc).__name__}: {str(exc)[:240]}"

    return row


# ---------------------------------------------------------------------------
# Main matrix
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="EGRET AOS driver (new paradigm, single-scenario)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--intensity", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--n-periods", type=int, default=4)
    parser.add_argument("--egret_file", type=str, default=None)
    parser.add_argument("--force-synthetic", action="store_true", default=True)
    parser.add_argument("--include-start-stop", action="store_true", default=False)
    parser.add_argument("--first-stage-names", type=str, default=None)
    parser.add_argument("--time-limit", type=float, default=180.0)
    parser.add_argument("--max-benders-iterations", type=int, default=80)
    parser.add_argument("--num-solutions", type=int, default=12)
    parser.add_argument("--summary-threshold", type=int, default=8)
    parser.add_argument(
        "--aos-methods", nargs="+", default=["gurobi_pool"],
        choices=["gurobi_pool", "binary"],
    )
    parser.add_argument(
        "--solve-modes", nargs="+", default=["extensive", "benders"],
        choices=["extensive", "benders"],
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fs_override = None
    if args.first_stage_names:
        fs_override = [s.strip() for s in args.first_stage_names.split(",") if s.strip()]

    rel_gaps = INTENSITY_GAPS[args.intensity]
    first_stage_modes = [False]
    if args.include_start_stop and fs_override is None:
        first_stage_modes.append(True)

    case = _case_label(args.egret_file, args.n_periods, args.force_synthetic)

    _print(f"Intensity {args.intensity} → rel_gaps = {rel_gaps}")
    _print(f"case = {case}")
    _print(f"n_periods = {args.n_periods}")
    _print(f"force_synthetic = {args.force_synthetic}")
    _print(f"aos_methods = {args.aos_methods}")
    _print(f"solve_modes = {args.solve_modes}")
    _print("Order: first_stage_mode → Baseline → aos_method → rel_gap → solve_mode")
    _print("Paradigm: public residual transform + solve_and_return_model (no monkey-patches)")

    rows: List[Dict[str, Any]] = []

    for include_ss in first_stage_modes:
        fs_label = "UnitOn+StartStop" if include_ss else "UnitOn"
        _print(f"\n===== first_stage_mode = {fs_label} =====")

        _print(f"Building SP  mode={fs_label} ...")
        sp, fs_names = build_egret_sp(
            n_periods=args.n_periods,
            include_start_stop=include_ss,
            egret_file=args.egret_file,
            force_synthetic=args.force_synthetic,
            first_stage_names=fs_override,
        )
        n_fs = len(
            getattr(sp, "int_to_FirstStageVar", {}).get(next(iter(sp.bundles)), {})
        )
        _print(f"  n_first_stage = {n_fs}, names = {fs_names}")

        # Baseline EF (also provides ef_reference for the single-scenario Benders rule)
        _print("Baseline extensive ...")
        ef_base = run_extensive_base(sp, time_limit=args.time_limit)
        _print(f"  EF status={ef_base['status']} obj={ef_base['obj_value']}")

        rows.append(
            run_one_config(
                n_periods=args.n_periods,
                include_start_stop=include_ss,
                solve_mode="extensive",
                aos_method="baseline",
                rel_gap=0.0,
                num_solutions=args.num_solutions,
                time_limit=args.time_limit,
                max_benders_iterations=args.max_benders_iterations,
                summary_threshold=args.summary_threshold,
                case=case,
                force_synthetic=args.force_synthetic,
                first_stage_names=fs_override,
                cached_ef_base=ef_base,
                sp=sp,
            )
        )

        # Baseline Benders (EF already done on this SP)
        benders_base = None
        if "benders" in args.solve_modes:
            _print("Baseline Benders (new paradigm, same SP after EF) ...")
            benders_base = run_benders_base(
                sp,
                time_limit=args.time_limit,
                max_iterations=args.max_benders_iterations,
                ef_reference=ef_base.get("ef_reference"),
            )
            _print(
                f"  Benders status={benders_base['status']} "
                f"obj={benders_base['obj_value']}"
            )
            rows.append(
                run_one_config(
                    n_periods=args.n_periods,
                    include_start_stop=include_ss,
                    solve_mode="benders",
                    aos_method="baseline",
                    rel_gap=0.0,
                    num_solutions=args.num_solutions,
                    time_limit=args.time_limit,
                    max_benders_iterations=args.max_benders_iterations,
                    summary_threshold=args.summary_threshold,
                    case=case,
                    force_synthetic=args.force_synthetic,
                    first_stage_names=fs_override,
                    cached_benders_base=benders_base,
                    cached_ef_base=ef_base,
                    sp=sp,
                )
            )

        # AOS matrix
        for aos_method in args.aos_methods:
            for gap in rel_gaps:
                for mode in args.solve_modes:
                    _print(f"  AOS  mode={mode} method={aos_method} gap={gap}")
                    rows.append(
                        run_one_config(
                            n_periods=args.n_periods,
                            include_start_stop=include_ss,
                            solve_mode=mode,
                            aos_method=aos_method,
                            rel_gap=gap,
                            num_solutions=args.num_solutions,
                            time_limit=args.time_limit,
                            max_benders_iterations=args.max_benders_iterations,
                            summary_threshold=args.summary_threshold,
                            case=case,
                            force_synthetic=args.force_synthetic,
                            first_stage_names=fs_override,
                            cached_ef_base=ef_base,
                            cached_benders_base=benders_base,
                            sp=sp,
                        )
                    )

    # Write CSV
    out_csv = out_dir / f"egret_aos_new_{case}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    _print(f"\nWrote {len(rows)} rows → {out_csv}")

    # Quick console summary
    _print("\n=== Summary ===")
    for r in rows:
        _print(
            f"  {r['first_stage_mode']:16s} {r['solve_mode']:10s} "
            f"{r['aos_method']:12s} gap={r['rel_gap']:<5} "
            f"status={r['status']:8s} obj={r['obj_value']} "
            f"n_true={r['n_true']}  {r.get('error_msg','')[:60]}"
        )


if __name__ == "__main__":
    main()
