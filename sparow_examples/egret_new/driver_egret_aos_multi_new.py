#!/usr/bin/env python3
"""
driver_egret_aos_multi_new.py
=============================
Multi-scenario EGRET AOS driver under the **new paradigm**.

Changes vs the original driver_egret_aos_multi.py
-------------------------------------------------
- Residual handling: locked public `egret_uc_subproblem_transform`.
- Master capture: `solve_and_return_model` → live upper_model.
- Multi-scenario operational rule:
    Build multi SP → EF a→ *rebuild* fresh multi SP → Benders

Usage (same directory as egret_sparow_staged_multi.py + stage8_multi_new_paradigm.py):

  python driver_egret_aos_multi_new.py --out_dir results_multi_new/ --intensity 1 \\
      --force-synthetic --n-periods 4 2>&1 | tee driver_multi_new.log
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
    "case", "n_periods", "n_scenarios", "first_stage_mode", "solve_mode",
    "aos_method", "rel_gap", "num_solutions_cap", "time_limit_s", "base_time_s",
    "aos_time_s", "total_time_s", "n_candidates", "n_true", "obj_value",
    "master_lb_or_eta", "n_cuts", "n_iterations", "n_first_stage", "status",
    "error_msg", "solutions_summary",
]

DEFAULT_SCENARIOS = [
    {"ID": "low",  "Probability": 0.25, "load_scale": 0.85},
    {"ID": "med",  "Probability": 0.50, "load_scale": 1.00},
    {"ID": "high", "Probability": 0.25, "load_scale": 1.15},
]


def _print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()


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


def _case_label(egret_file, n_periods, force_synthetic, n_scen) -> str:
    if force_synthetic or egret_file is None:
        return f"synthetic_T{n_periods if n_periods else 4}_S{n_scen}"
    return f"egret:{Path(egret_file).stem}_T{n_periods if n_periods else 'full'}_S{n_scen}"


# ---------------------------------------------------------------------------
# Multi SP build
# ---------------------------------------------------------------------------
def build_egret_sp_multi(
    n_periods: Optional[int] = 4,
    include_start_stop: bool = False,
    egret_file: Optional[str] = None,
    force_synthetic: bool = True,
    first_stage_names: Optional[Sequence[str]] = None,
    scenarios: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Any, List[str], Any, List[Dict[str, Any]]]:
    """
    Returns (sp, first_stage_names, base_md, scenarios).
    base_md is retained so the multi rule can rebuild a fresh SP after EF.
    """
    from egret_sparow_staged_multi import (
        stage0_check_environment,
        stage1_build_egret_model,
        stage2_inspect_variables,
        stage3_verify_first_stage_names,
        stage5_egret_sparow_sp_multi,
    )

    stage0_check_environment()
    use_synthetic = bool(force_synthetic) or (egret_file is None)

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

    if scenarios is None:
        scenarios = list(DEFAULT_SCENARIOS)
    _print(f"  scenarios = {scenarios}")

    sp = stage5_egret_sparow_sp_multi(
        md, first_stage_names=first_stage_names, scenarios=scenarios
    )
    return sp, list(first_stage_names), md, scenarios


def rebuild_fresh_multi_sp(md, first_stage_names, scenarios):
    from egret_sparow_staged_multi import stage5_egret_sparow_sp_multi
    _print("  Rebuilding multi-scenario SP for Benders (EF may have altered bundles) ...")
    return stage5_egret_sparow_sp_multi(
        md, first_stage_names=first_stage_names, scenarios=scenarios
    )


# ---------------------------------------------------------------------------
# Baseline solves
# ---------------------------------------------------------------------------
def run_extensive_base(sp, time_limit: float) -> Dict[str, Any]:
    from egret_sparow_staged_multi import stage6_solve_extensive_form

    t0 = time.perf_counter()
    try:
        ef_ref = stage6_solve_extensive_form(
            sp, solver_name="gurobi", time_limit=time_limit
        )
        elapsed = time.perf_counter() - t0
        return {
            "model": None,
            "ef_reference": ef_ref,
            "base_time_s": elapsed,
            "obj_value": float(ef_ref["objective"]),
            "master_lb_or_eta": float(ef_ref["objective"]),
            "n_cuts": float("nan"),
            "n_iterations": float("nan"),
            "status": "optimal",
            "error_msg": "",
        }
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


def run_benders_base_multi(
    sp_fresh,
    ef_reference: dict,
    time_limit: float,
    max_iterations: int = 80,
    eta_lower: float = 0.0,
) -> Dict[str, Any]:
    """
    New-paradigm multi Benders base.
    Caller must supply a *fresh* SP (post-EF rebuild) and the EF reference.
    """
    from stage8_multi_new_paradigm import stage8_run_benders_multi

    t0 = time.perf_counter()
    try:
        out = stage8_run_benders_multi(
            sp_fresh,
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
# AOS-Benders (same as single)
# ---------------------------------------------------------------------------
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
# One configuration (simplified multi: baseline + AOS-Benders focus)
# ---------------------------------------------------------------------------
def run_one_config(
    *,
    n_periods: Optional[int],
    n_scenarios: int,
    include_start_stop: bool,
    solve_mode: str,
    aos_method: str,
    rel_gap: float,
    num_solutions: int,
    time_limit: float,
    max_benders_iterations: int,
    summary_threshold: int,
    case: str,
    cached_ef_base: Optional[Dict[str, Any]] = None,
    cached_benders_base: Optional[Dict[str, Any]] = None,
    sp: Any = None,
) -> Dict[str, Any]:
    fs_label = "UnitOn+StartStop" if include_start_stop else "UnitOn"
    row: Dict[str, Any] = {
        "case": case,
        "n_periods": n_periods if n_periods is not None else float("nan"),
        "n_scenarios": n_scenarios,
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
                raise RuntimeError("multi extensive requires cached_ef_base")
            row["n_first_stage"] = (
                len(getattr(sp, "int_to_FirstStageVar", {}).get(next(iter(sp.bundles)), {}))
                if sp is not None else float("nan")
            )
            row["base_time_s"] = base["base_time_s"]
            row["obj_value"] = base["obj_value"]
            row["master_lb_or_eta"] = base["master_lb_or_eta"]
            row["status"] = base["status"]
            row["error_msg"] = base.get("error_msg", "")
            if aos_method == "baseline":
                row["n_candidates"] = 1
                row["n_true"] = 1
                row["aos_time_s"] = 0.0
                row["solutions_summary"] = "baseline"
            else:
                # Multi EF AOS is left as a future extension; mark skipped
                row["status"] = "skipped"
                row["error_msg"] = "multi EF AOS not yet ported in new-paradigm driver"
                row["aos_time_s"] = 0.0

        else:  # benders
            base = cached_benders_base
            if base is None:
                raise RuntimeError("multi benders requires cached_benders_base")
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
                upper = base.get("model")
                if upper is None:
                    row["status"] = "error"
                    row["error_msg"] = "no upper_model from multi Benders base"
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
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="EGRET AOS driver (new paradigm, multi-scenario)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--intensity", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--n-periods", type=int, default=4)
    parser.add_argument("--egret_file", type=str, default=None)
    parser.add_argument("--force-synthetic", action="store_true", default=True)
    parser.add_argument("--include-start-stop", action="store_true", default=False)
    parser.add_argument("--first-stage-names", type=str, default=None)
    parser.add_argument("--time-limit", type=float, default=300.0)
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
    parser.add_argument(
        "--scales", type=str, default="0.85,1.0,1.15",
        help="Comma-separated load scales",
    )
    parser.add_argument(
        "--probs", type=str, default="0.25,0.5,0.25",
        help="Comma-separated probabilities (must sum to 1)",
    )
    parser.add_argument(
        "--scenario-ids", type=str, default="low,med,high",
        help="Comma-separated scenario IDs",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fs_override = None
    if args.first_stage_names:
        fs_override = [s.strip() for s in args.first_stage_names.split(",") if s.strip()]

    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    probs = [float(x) for x in args.probs.split(",") if x.strip()]
    ids = [s.strip() for s in args.scenario_ids.split(",") if s.strip()]
    if not (len(scales) == len(probs) == len(ids)):
        parser.error("--scales, --probs, --scenario-ids must have same length")
    if abs(sum(probs) - 1.0) > 1e-8:
        parser.error(f"--probs must sum to 1 (got {sum(probs)})")
    scenarios = [
        {"ID": sid, "Probability": p, "load_scale": sc}
        for sid, p, sc in zip(ids, probs, scales)
    ]

    rel_gaps = INTENSITY_GAPS[args.intensity]
    case = _case_label(args.egret_file, args.n_periods, args.force_synthetic, len(scenarios))

    _print(f"Intensity {args.intensity} → rel_gaps = {rel_gaps}")
    _print(f"case = {case}")
    _print(f"scenarios = {scenarios}")
    _print(f"aos_methods = {args.aos_methods}")
    _print(f"solve_modes = {args.solve_modes}")
    _print("Paradigm: public residual transform + solve_and_return_model")
    _print("Multi rule: EF → *fresh SP rebuild* → Benders")

    rows: List[Dict[str, Any]] = []
    include_ss = bool(args.include_start_stop)
    fs_label = "UnitOn+StartStop" if include_ss else "UnitOn"
    _print(f"\n===== first_stage_mode = {fs_label} =====")

    # Build multi SP + keep md for rebuild
    sp, fs_names, base_md, scenarios = build_egret_sp_multi(
        n_periods=args.n_periods,
        include_start_stop=include_ss,
        egret_file=args.egret_file,
        force_synthetic=args.force_synthetic,
        first_stage_names=fs_override,
        scenarios=scenarios,
    )
    n_fs = len(
        getattr(sp, "int_to_FirstStageVar", {}).get(next(iter(sp.bundles)), {})
    )
    _print(f"  n_first_stage = {n_fs}, names = {fs_names}")

    # Baseline EF
    _print("Baseline multi extensive ...")
    ef_base = run_extensive_base(sp, time_limit=args.time_limit)
    _print(f"  EF status={ef_base['status']} obj={ef_base['obj_value']}")

    rows.append(
        run_one_config(
            n_periods=args.n_periods,
            n_scenarios=len(scenarios),
            include_start_stop=include_ss,
            solve_mode="extensive",
            aos_method="baseline",
            rel_gap=0.0,
            num_solutions=args.num_solutions,
            time_limit=args.time_limit,
            max_benders_iterations=args.max_benders_iterations,
            summary_threshold=args.summary_threshold,
            case=case,
            cached_ef_base=ef_base,
            sp=sp,
        )
    )

    # Multi rule: rebuild fresh SP, then Benders
    benders_base = None
    sp_benders = sp
    if "benders" in args.solve_modes and ef_base["status"] == "optimal":
        sp_benders = rebuild_fresh_multi_sp(base_md, fs_names, scenarios)
        _print("Baseline multi Benders (new paradigm, fresh SP) ...")
        benders_base = run_benders_base_multi(
            sp_benders,
            ef_reference=ef_base["ef_reference"],
            time_limit=args.time_limit,
            max_iterations=args.max_benders_iterations,
        )
        _print(
            f"  Benders status={benders_base['status']} "
            f"obj={benders_base['obj_value']}"
        )
        rows.append(
            run_one_config(
                n_periods=args.n_periods,
                n_scenarios=len(scenarios),
                include_start_stop=include_ss,
                solve_mode="benders",
                aos_method="baseline",
                rel_gap=0.0,
                num_solutions=args.num_solutions,
                time_limit=args.time_limit,
                max_benders_iterations=args.max_benders_iterations,
                summary_threshold=args.summary_threshold,
                case=case,
                cached_ef_base=ef_base,
                cached_benders_base=benders_base,
                sp=sp_benders,
            )
        )

    # AOS matrix (Benders path is the primary focus for multi)
    for aos_method in args.aos_methods:
        for gap in rel_gaps:
            for mode in args.solve_modes:
                _print(f"  AOS  mode={mode} method={aos_method} gap={gap}")
                rows.append(
                    run_one_config(
                        n_periods=args.n_periods,
                        n_scenarios=len(scenarios),
                        include_start_stop=include_ss,
                        solve_mode=mode,
                        aos_method=aos_method,
                        rel_gap=gap,
                        num_solutions=args.num_solutions,
                        time_limit=args.time_limit,
                        max_benders_iterations=args.max_benders_iterations,
                        summary_threshold=args.summary_threshold,
                        case=case,
                        cached_ef_base=ef_base,
                        cached_benders_base=benders_base,
                        sp=sp_benders if mode == "benders" else sp,
                    )
                )

    out_csv = out_dir / f"egret_aos_multi_new_{case}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    _print(f"\nWrote {len(rows)} rows → {out_csv}")

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
