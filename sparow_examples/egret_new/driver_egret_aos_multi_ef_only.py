#!/usr/bin/env python3
"""
driver_egret_aos_multi_ef_only.py
=================================
Multi-scenario EGRET **EF + EF AOS only** driver.

Path (no double solve):
  1. Build multi SP
  2. model = sp.create_EF(compact_repn=True)  # fallback False if create fails
  3. or_topas.aos.gurobi_generate_solutions(model, ...) at each rel_gap

Usage (same directory as egret_sparow_staged_multi.py):

  python driver_egret_aos_multi_ef_only.py --out_dir results_ef_multi/ --intensity 3 \\
      --force-synthetic --n-periods 4 2>&1 | tee driver_multi_ef_only.log
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
    2: [0.0, 0.01],
    3: [0.0, 0.01, 0.05],
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
    except Exception:
        pass
    return None


def _extract_uniton_compact(sol_or_model) -> str:
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


def build_egret_sp_multi(
    n_periods: Optional[int] = 4,
    include_start_stop: bool = False,
    egret_file: Optional[str] = None,
    force_synthetic: bool = True,
    first_stage_names: Optional[Sequence[str]] = None,
    scenarios: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Any, List[str], List[Dict[str, Any]]]:
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
    return sp, list(first_stage_names), scenarios


def create_ef_model(sp, compact: bool = True):
    """Build EF model only — no solve. Falls back to non-compact if needed."""
    try:
        model = sp.create_EF(compact_repn=compact)
        _print(f"  create_EF(compact_repn={compact}) OK → {type(model).__name__}")
        return model, compact
    except Exception as exc:
        if compact:
            _print(
                f"  create_EF(compact_repn=True) failed: {type(exc).__name__}: {exc}"
            )
            _print("  Retrying create_EF(compact_repn=False) ...")
            model = sp.create_EF(compact_repn=False)
            _print(f"  create_EF(compact_repn=False) OK → {type(model).__name__}")
            return model, False
        raise


def run_aos_extensive(
    model,
    aos_method: str,
    rel_gap: float,
    num_solutions: int,
    time_limit: float,
    keep_pool: bool = True,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    if model is None:
        return {
            "aos_time_s": 0.0,
            "n_candidates": float("nan"),
            "n_true": float("nan"),
            "obj_value": float("nan"),
            "status": "error",
            "error_msg": "no EF model",
            "pool": None,
        }

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
            raise ValueError(f"Unknown aos_method: {aos_method}")
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
    status = "timeout" if elapsed > time_limit else "optimal"
    n = len(pool) if pool is not None else 0
    objs = []
    if pool:
        for sol in pool:
            o = _sol_objective(sol)
            if o is not None:
                objs.append(o)
    best = min(objs) if objs else float("nan")
    return {
        "aos_time_s": elapsed,
        "n_candidates": n,
        "n_true": n,
        "obj_value": best,
        "status": status,
        "error_msg": "",
        "pool": pool if keep_pool else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Multi EF + EF AOS only (create_EF → gurobi_pool, no prior solve)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--intensity", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--n-periods", type=int, default=4)
    parser.add_argument("--egret_file", type=str, default=None)
    parser.add_argument("--force-synthetic", action="store_true", default=False,
                        help="Ignore --egret_file and use synthetic tiny UC")
    parser.add_argument("--include-start-stop", action="store_true", default=False)
    parser.add_argument("--first-stage-names", type=str, default=None)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--num-solutions", type=int, default=12)
    parser.add_argument("--summary-threshold", type=int, default=8)
    parser.add_argument(
        "--aos-methods", nargs="+", default=["gurobi_pool"],
        choices=["gurobi_pool", "binary"],
    )
    parser.add_argument("--scales", type=str, default="0.85,1.0,1.15")
    parser.add_argument("--probs", type=str, default="0.25,0.5,0.25")
    parser.add_argument("--scenario-ids", type=str, default="low,med,high")
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
    include_ss = bool(args.include_start_stop)
    fs_label = "UnitOn+StartStop" if include_ss else "UnitOn"

    _print(f"Intensity {args.intensity} → rel_gaps = {rel_gaps}")
    _print(f"case = {case}")
    _print(f"scenarios = {scenarios}")
    _print("Mode: create_EF → gurobi_generate_solutions (no prior EF solve)")

    rows: List[Dict[str, Any]] = []

    _print(f"\n===== first_stage_mode = {fs_label} =====")
    sp, fs_names, scenarios = build_egret_sp_multi(
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

    t_build = time.perf_counter()
    try:
        model, used_compact = create_ef_model(sp, compact=True)
    except Exception as exc:
        _print(f"  FATAL: could not create EF model: {exc}")
        return
    build_s = time.perf_counter() - t_build
    _print(f"  EF build time = {build_s:.3f}s  compact={used_compact}")

    baseline_obj = float("nan")

    for aos_method in args.aos_methods:
        for gap in rel_gaps:
            _print(f"  EF AOS  method={aos_method} gap={gap}")
            aos = run_aos_extensive(
                model, aos_method, gap, args.num_solutions, args.time_limit
            )
            if baseline_obj != baseline_obj and aos["obj_value"] == aos["obj_value"]:
                baseline_obj = aos["obj_value"]
                _print(f"  baseline obj (from gap={gap} pool) = {baseline_obj}")

            rows.append({
                "case": case,
                "n_periods": args.n_periods,
                "n_scenarios": len(scenarios),
                "first_stage_mode": fs_label,
                "solve_mode": "extensive",
                "aos_method": aos_method,
                "rel_gap": gap,
                "num_solutions_cap": args.num_solutions,
                "time_limit_s": args.time_limit,
                "base_time_s": build_s,
                "aos_time_s": aos["aos_time_s"],
                "total_time_s": build_s + float(
                    aos["aos_time_s"] if aos["aos_time_s"] == aos["aos_time_s"] else 0
                ),
                "n_candidates": aos["n_candidates"],
                "n_true": aos["n_true"],
                "obj_value": aos["obj_value"],
                "master_lb_or_eta": baseline_obj,
                "n_cuts": float("nan"),
                "n_iterations": float("nan"),
                "n_first_stage": n_fs,
                "status": aos["status"],
                "error_msg": aos.get("error_msg", ""),
                "solutions_summary": _format_solution_summary(
                    aos.get("pool"), args.summary_threshold
                ),
            })

    if baseline_obj == baseline_obj:
        rows.insert(0, {
            "case": case,
            "n_periods": args.n_periods,
            "n_scenarios": len(scenarios),
            "first_stage_mode": fs_label,
            "solve_mode": "extensive",
            "aos_method": "baseline",
            "rel_gap": 0.0,
            "num_solutions_cap": args.num_solutions,
            "time_limit_s": args.time_limit,
            "base_time_s": build_s,
            "aos_time_s": 0.0,
            "total_time_s": build_s,
            "n_candidates": 1,
            "n_true": 1,
            "obj_value": baseline_obj,
            "master_lb_or_eta": baseline_obj,
            "n_cuts": float("nan"),
            "n_iterations": float("nan"),
            "n_first_stage": n_fs,
            "status": "optimal",
            "error_msg": "",
            "solutions_summary": "baseline (from AOS pool)",
        })

    out_csv = out_dir / f"egret_aos_multi_ef_only_{case}.csv"
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
