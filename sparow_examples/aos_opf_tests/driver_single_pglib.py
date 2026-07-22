#!/usr/bin/env python
"""
PGLIB-OPF AOS-Benders single-case benchmark driver (v1.3.11)

Assumes Gurobi is installed, licensed, and available.

Ordering
--------
commitment is the outermost axis:
  1. every configuration with commitment=False
  2. then every configuration with commitment=True

Inside each commitment block the order is:
  fidelity → [Baseline extensive] → aos_method → rel_gap → solve_mode

so that:
  - a pure extensive-form baseline appears first for each fidelity
    (labelled "0/Baseline", "1/Baseline", "2/Baseline"),
  - then the extensive-form and Benders AOS runs for the same
    (commitment, fidelity, aos_method, gap) sit next to each other
    in the CSV (easy side-by-side comparison).

By default, when commitment=False the driver only runs copper-plate
(fidelity=0).  This keeps the extensive-form vs Benders n_true counts
comparable on the first-stage generation variables.  Network models
(fidelity 1/2) introduce projected-out recourse variables that make the
two methods count different numbers of solutions.  Pass
--all_fidelities_for_continuous to restore the original full matrix for
the continuous arm.  Commitment=True always runs all three fidelities.

--intensity still controls the relative-gap levels (default = 1 → [0.0]).

When the number of true solutions is ≤ --summary_threshold (default 5)
a compact representation of each solution (via pyomo_utils.simplify_solution)
is written into the solutions_summary column.  Solutions are separated by
" || " and fields inside a solution by ";".

Usage
-----
python driver_single_pglib.py \\
    --pglib_file pglib_opf_case24_ieee_rts.m \\
    --out_dir   . \\
    [--intensity 1] [--time_limit 100] [--epsilon 1e-4] [--max_sols 50] \\
    [--all_fidelities_for_continuous] [--summary_threshold 5]
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pyomo.environ as pyo

from or_topas.benders import BendersGenerator_Serial as BendersCutGenerator
import or_topas.benders.tests.test_cases as tc
from or_topas.benders.aos_benders import (
    aos_benders_generate_candidates,
    aos_benders_filter,
)
from or_topas.aos import (
    enumerate_linear_solutions,
    enumerate_binary_solutions,
    gurobi_generate_solutions,
)
from or_topas.util.pyomo_utils import simplify_solution

# Hard-coded solver names (Gurobi assumed available)
BENDERS_SOLVER = "gurobi_persistent"
AOS_SOLVER = "gurobi"

# Relative-gap levels by intensity (always sorted ascending)
INTENSITY_GAPS = {
    1: [0.0],
    2: [0.0, 0.01],
    3: [0.0, 0.01, 0.05],
    4: [0.0, 0.01, 0.05, 0.10],
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def build_grid(case_path: str, commitment: bool, epsilon: float):
    if commitment:
        return tc.MatpowerGridWithCommitment(m_file=case_path, epsilon=epsilon)
    return tc.MatpowerGrid(m_file=case_path)


def _set_time_limit(opt, time_limit: float) -> None:
    try:
        opt.options["TimeLimit"] = time_limit
    except Exception:
        pass
    try:
        opt.options["timelimit"] = time_limit
    except Exception:
        pass


def _format_solution_summary(pool, threshold: int) -> str:
    """Return a compact, CSV-safe string of the solutions when ≤ threshold."""
    if pool is None:
        return ""
    try:
        n = len(pool)
    except TypeError:
        return ""
    if n == 0:
        return ""
    if n > threshold:
        return f"too_many ({n})"

    parts = []
    for sol in pool:
        try:
            d = simplify_solution(sol)
            fields = []
            for key, (name, val) in d.items():
                if key == "id":
                    continue
                label = name if name else key
                if isinstance(val, float):
                    fields.append(f"{label}={val:.6g}")
                else:
                    fields.append(f"{label}={val}")
            parts.append(";".join(fields))
        except Exception:
            parts.append("simplify_failed")
    return " || ".join(parts)


# ---------------------------------------------------------------------------
# Extensive-form base solve
# ---------------------------------------------------------------------------

def run_extensive(
    grid,
    fidelity: int,
    commitment: bool,
    time_limit: float,
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    if commitment:
        model = tc.EnergyGridWithCommitment.create_tiny_opf(grid, mode=fidelity)
    else:
        model = tc.EnergyGrid.create_tiny_opf(grid, mode=fidelity)

    opt = pyo.SolverFactory(BENDERS_SOLVER)
    _set_time_limit(opt, time_limit)
    opt.set_instance(model)

    try:
        res = opt.solve(tee=False, save_results=False)
        term = res.solver.termination_condition
        if term == pyo.TerminationCondition.optimal:
            status = "optimal"
        elif term in (
            pyo.TerminationCondition.maxTimeLimit,
            pyo.TerminationCondition.maxIterations,
        ):
            status = "timeout"
        else:
            status = str(term)
        obj = pyo.value(model.obj)
    except Exception:
        status = "error"
        obj = float("nan")
        raise

    elapsed = time.perf_counter() - t0
    if elapsed > time_limit + 1.0:
        status = "timeout"

    return {
        "model": model,
        "base_time_s": elapsed,
        "obj_value": obj,
        "master_lb_or_eta": obj,
        "n_cuts": 0,
        "n_iterations": 1,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Benders base solve
# ---------------------------------------------------------------------------

def run_benders(
    grid,
    fidelity: int,
    commitment: bool,
    time_limit: float,
    eta_lb: float = -1e5,
    eta_ub: float = 1e5,
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    if commitment:
        if hasattr(tc.EnergyGridWithCommitment, "setup_energy_grid_commitment_persistent"):
            opt, m = tc.EnergyGridWithCommitment.setup_energy_grid_commitment_persistent(
                solver_name=BENDERS_SOLVER,
                grid=grid,
                eta_lb=eta_lb,
                eta_ub=eta_ub,
                mode=fidelity,
            )
        else:
            m = tc.EnergyGridWithCommitment.create_root(grid, eta_lb=eta_lb, eta_ub=eta_ub)
            root_vars = list(m.commit.values())
            m.benders = BendersCutGenerator()
            m.benders.set_input(
                root_vars=root_vars,
                tol=1e-8,
                transform="standard_lp",
                allow_infeasible=True,
            )
            m.benders.add_subproblem(
                subproblem_fn=tc.EnergyGridWithCommitment.create_subproblem,
                subproblem_fn_kwargs={"root": m, "grid": grid, "mode": fidelity},
                root_eta=m.eta,
                subproblem_solver=BENDERS_SOLVER,
            )
            opt = pyo.SolverFactory(BENDERS_SOLVER)
            opt.set_instance(m)
    else:
        m = tc.EnergyGrid.create_root(grid)
        m.eta.setlb(eta_lb)
        m.eta.setub(eta_ub)
        root_vars = list(m.generation.values())
        m.benders = BendersCutGenerator()
        m.benders.set_input(
            root_vars=root_vars,
            tol=1e-8,
            transform="standard_lp",
            allow_infeasible=True,
        )
        m.benders.add_subproblem(
            subproblem_fn=tc.EnergyGrid.create_subproblem,
            subproblem_fn_kwargs={"root": m, "grid": grid, "mode": fidelity},
            root_eta=m.eta,
            subproblem_solver=BENDERS_SOLVER,
        )
        opt = pyo.SolverFactory(BENDERS_SOLVER)
        opt.set_instance(m)

    _set_time_limit(opt, time_limit)

    n_cuts_total = 0
    n_iters = 0
    max_iters = 80
    status = "optimal"

    for i in range(max_iters):
        if time.perf_counter() - t0 > time_limit:
            status = "timeout"
            break
        opt.solve(tee=False, save_results=False)
        cuts = m.benders.generate_cut()
        for c in cuts:
            opt.add_constraint(c)
        n_cuts_total += len(cuts)
        n_iters = i + 1
        if not cuts:
            break
    else:
        if status != "timeout":
            status = "max_iter"

    elapsed = time.perf_counter() - t0
    final_eta = pyo.value(m.eta) if m.eta.value is not None else float("nan")

    return {
        "model": m,
        "opt": opt,
        "base_time_s": elapsed,
        "n_cuts": n_cuts_total,
        "n_iterations": n_iters,
        "master_lb_or_eta": final_eta,
        "status": status,
    }


# ---------------------------------------------------------------------------
# AOS on extensive form (direct generation, no filter)
# ---------------------------------------------------------------------------

def run_aos_extensive(
    model,
    aos_method: str,
    rel_gap: float,
    num_solutions: int,
    time_limit: float,
    keep_pool: bool = False,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    status = "optimal"
    pool = None

    try:
        if aos_method == "linear":
            pool = enumerate_linear_solutions(
                model,
                num_solutions=num_solutions,
                rel_opt_gap=rel_gap,
                solver=AOS_SOLVER,
                tee=False,
            )
        elif aos_method == "binary":
            pool = enumerate_binary_solutions(
                model,
                num_solutions=num_solutions,
                rel_opt_gap=rel_gap,
                solver=AOS_SOLVER,
                tee=False,
            )
        elif aos_method == "gurobi_pool":
            pool = gurobi_generate_solutions(
                model,
                num_solutions=num_solutions,
                rel_opt_gap=rel_gap,
                pool_search_mode=2,
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

    n = len(pool)
    return {
        "aos_time_s": elapsed,
        "n_candidates": n,
        "n_true": n,
        "obj_value": float("nan"),
        "status": status,
        "pool": pool if keep_pool else None,
    }


# ---------------------------------------------------------------------------
# AOS on Benders master (generate + true-recourse filter)
# ---------------------------------------------------------------------------

def run_aos_benders(
    m,
    aos_method: str,
    rel_gap: float,
    num_solutions: int,
    time_limit: float,
    keep_pool: bool = False,
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    enum_map = {
        "linear": "linear",
        "binary": "binary",
        "gurobi_pool": "gurobi_pool",
    }
    enumeration_method = enum_map[aos_method]

    try:
        candidate_pool, data = aos_benders_generate_candidates(
            m=m,
            rel_gap=rel_gap,
            num_solutions=num_solutions,
            mip_solver=AOS_SOLVER,
            enumeration_method=enumeration_method,
            tee=False,
        )
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

    if time.perf_counter() - t0 > time_limit:
        return {
            "aos_time_s": time.perf_counter() - t0,
            "n_candidates": len(candidate_pool),
            "n_true": float("nan"),
            "obj_value": float("nan"),
            "status": "timeout",
            "pool": None,
        }

    true_pool = aos_benders_filter(candidate_pool, data, tee=False, tee_final=False)

    elapsed = time.perf_counter() - t0
    status = "timeout" if elapsed > time_limit else "optimal"

    return {
        "aos_time_s": elapsed,
        "n_candidates": len(candidate_pool),
        "n_true": len(true_pool),
        "obj_value": data.lower_bound,
        "status": status,
        "pool": true_pool if keep_pool else None,
    }


# ---------------------------------------------------------------------------
# One configuration (normal AOS matrix entry)
# ---------------------------------------------------------------------------

def run_one_config(
    case_path: str,
    fidelity: int,
    commitment: bool,
    solve_mode: str,
    aos_method: str,
    rel_gap: float,
    num_solutions: int,
    epsilon: float,
    time_limit: float,
    summary_threshold: int = 5,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "case": Path(case_path).name,
        "grid_fidelity": fidelity,
        "commitment": commitment,
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
        "n_buses": float("nan"),
        "n_lines": float("nan"),
        "n_gens_or_commits": float("nan"),
        "status": "error",
        "error_msg": "",
        "solutions_summary": "",
    }

    try:
        grid = build_grid(case_path, commitment, epsilon)
        row["n_buses"] = len(grid.buses)
        row["n_lines"] = len(grid.lines)
        if commitment:
            row["n_gens_or_commits"] = len(getattr(grid, "gen_buses", []))
        else:
            row["n_gens_or_commits"] = sum(
                1 for v in grid.gen_max_dict.values() if v > 0
            )

        # ---- base solve ----
        if solve_mode == "extensive":
            base = run_extensive(
                grid=grid,
                fidelity=fidelity,
                commitment=commitment,
                time_limit=time_limit,
            )
        else:
            base = run_benders(
                grid=grid,
                fidelity=fidelity,
                commitment=commitment,
                time_limit=time_limit,
            )

        row["base_time_s"] = base["base_time_s"]
        row["n_cuts"] = base.get("n_cuts", 0)
        row["n_iterations"] = base.get("n_iterations", 1)
        row["master_lb_or_eta"] = base["master_lb_or_eta"]
        row["status"] = base["status"]
        row["obj_value"] = base.get("obj_value", base["master_lb_or_eta"])

        # ---- AOS phase ----
        keep_pool = True

        if base["status"] not in ("error",):
            if solve_mode == "extensive":
                aos_res = run_aos_extensive(
                    model=base["model"],
                    aos_method=aos_method,
                    rel_gap=rel_gap,
                    num_solutions=num_solutions,
                    time_limit=time_limit,
                    keep_pool=keep_pool,
                )
            else:
                aos_res = run_aos_benders(
                    m=base["model"],
                    aos_method=aos_method,
                    rel_gap=rel_gap,
                    num_solutions=num_solutions,
                    time_limit=time_limit,
                    keep_pool=keep_pool,
                )

            row["aos_time_s"] = aos_res["aos_time_s"]
            row["n_candidates"] = aos_res["n_candidates"]
            row["n_true"] = aos_res["n_true"]
            if not math.isnan(aos_res.get("obj_value", float("nan"))):
                row["obj_value"] = aos_res["obj_value"]
            if aos_res["status"] in ("timeout", "error"):
                row["status"] = aos_res["status"]
            if "error_msg" in aos_res:
                row["error_msg"] = aos_res["error_msg"]

            row["solutions_summary"] = _format_solution_summary(
                aos_res.get("pool"), summary_threshold
            )

        row["total_time_s"] = (
            (row["base_time_s"] if not math.isnan(row["base_time_s"]) else 0.0)
            + (row["aos_time_s"] if not math.isnan(row["aos_time_s"]) else 0.0)
        )

    except Exception as exc:
        row["status"] = "error"
        row["error_msg"] = f"{type(exc).__name__}: {str(exc)[:300]}"

    return row


# ---------------------------------------------------------------------------
# Pure extensive-form baseline (no AOS) for a given fidelity
# ---------------------------------------------------------------------------

def run_baseline_extensive(
    case_path: str,
    fidelity: int,
    commitment: bool,
    epsilon: float,
    time_limit: float,
) -> Dict[str, Any]:
    """Pure extensive-form solve used as an accuracy reference for a fidelity."""
    row: Dict[str, Any] = {
        "case": Path(case_path).name,
        "grid_fidelity": f"{fidelity}/Baseline",
        "commitment": commitment,
        "solve_mode": "extensive",
        "aos_method": "baseline",
        "rel_gap": 0.0,
        "num_solutions_cap": 1,
        "time_limit_s": time_limit,
        "base_time_s": float("nan"),
        "aos_time_s": 0.0,
        "total_time_s": float("nan"),
        "n_candidates": 1,
        "n_true": 1,
        "obj_value": float("nan"),
        "master_lb_or_eta": float("nan"),
        "n_cuts": 0,
        "n_iterations": 1,
        "n_buses": float("nan"),
        "n_lines": float("nan"),
        "n_gens_or_commits": float("nan"),
        "status": "error",
        "error_msg": "",
        "solutions_summary": "",
    }

    try:
        grid = build_grid(case_path, commitment, epsilon)
        row["n_buses"] = len(grid.buses)
        row["n_lines"] = len(grid.lines)
        if commitment:
            row["n_gens_or_commits"] = len(getattr(grid, "gen_buses", []))
        else:
            row["n_gens_or_commits"] = sum(
                1 for v in grid.gen_max_dict.values() if v > 0
            )

        base = run_extensive(
            grid=grid,
            fidelity=fidelity,
            commitment=commitment,
            time_limit=time_limit,
        )

        row["base_time_s"] = base["base_time_s"]
        row["total_time_s"] = base["base_time_s"]
        row["obj_value"] = base["obj_value"]
        row["master_lb_or_eta"] = base["master_lb_or_eta"]
        row["status"] = base["status"]
        row["n_cuts"] = 0
        row["n_iterations"] = 1
        row["solutions_summary"] = f"obj={base['obj_value']}"

    except Exception as exc:
        row["status"] = "error"
        row["error_msg"] = f"{type(exc).__name__}: {str(exc)[:300]}"

    return row


# ---------------------------------------------------------------------------
# Main – commitment outermost, smart gap pruning, intensity control
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PGLIB-OPF AOS-Benders single-case driver (v1.3.11) – commitment outermost"
    )
    parser.add_argument("--pglib_file", required=True, help="Path to one .m file")
    parser.add_argument("--out_dir", required=True, help="Directory for the results CSV")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-4,
        help=(
            "Minimum generation when a unit is committed "
            "(generation >= epsilon * commit). Default 1e-4."
        ),
    )
    parser.add_argument("--max_sols", type=int, default=50, help="num_solutions cap (no hard upper bound)")
    parser.add_argument(
        "--time_limit",
        type=float,
        default=100.0,
        help="Wall-clock limit (seconds) per phase",
    )
    parser.add_argument(
        "--intensity",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help=(
            "How many relative-gap levels to run (always in increasing order). "
            "1=[0.0], 2=[0.0,0.01], 3=[0.0,0.01,0.05], 4=[0.0,0.01,0.05,0.10]"
        ),
    )
    parser.add_argument(
        "--all_fidelities_for_continuous",
        action="store_true",
        default=False,
        help=(
            "If set, run grid_fidelity 0/1/2 also for commitment=False. "
            "Default (False) restricts continuous models to copper-plate (fidelity=0) only, "
            "so that extensive vs Benders solution counts remain comparable on the first-stage "
            "generation variables. Network models introduce projected-out recourse variables "
            "that make the counts diverge. Commitment=True always runs all three fidelities."
        ),
    )
    parser.add_argument(
        "--summary_threshold",
        type=int,
        default=5,
        help=(
            "When the number of true solutions is ≤ this value, write a compact "
            "representation of each solution (via pyomo_utils.simplify_solution) "
            "into the solutions_summary column. Default 5."
        ),
    )
    args = parser.parse_args()

    case_path = Path(args.pglib_file)
    if not case_path.is_file():
        raise FileNotFoundError(f"PGLIB file not found: {case_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Experimental matrix ----
    commitments = [False, True]          # outermost – all False first, then all True
    rel_gaps = INTENSITY_GAPS[args.intensity]
    num_solutions = args.max_sols
    time_limit = args.time_limit
    summary_threshold = args.summary_threshold

    print(f"Intensity {args.intensity} → rel_gaps = {rel_gaps}")
    print(f"summary_threshold = {summary_threshold}")
    print("Execution order: all commitment=False, then all commitment=True")
    print("  (within each: fidelity → Baseline → aos_method → rel_gap → solve_mode)")
    if not args.all_fidelities_for_continuous:
        print("Continuous models restricted to copper-plate (fidelity=0) only "
              "(pass --all_fidelities_for_continuous to override)")

    rows: List[Dict[str, Any]] = []
    hit_cap: Dict[Tuple, bool] = {}

    # Pre-compute size columns once per commitment flavour so they never become NaN
    size_cache: Dict[bool, Dict[str, int]] = {}
    for _c in (False, True):
        try:
            _g = build_grid(str(case_path), _c, args.epsilon)
            size_cache[_c] = {
                "n_buses": len(_g.buses),
                "n_lines": len(_g.lines),
                "n_gens_or_commits": (
                    len(getattr(_g, "gen_buses", []))
                    if _c
                    else sum(1 for v in _g.gen_max_dict.values() if v > 0)
                ),
            }
        except Exception:
            size_cache[_c] = {
                "n_buses": float("nan"),
                "n_lines": float("nan"),
                "n_gens_or_commits": float("nan"),
            }

    for commitment in commitments:                 # <-- outermost
        # Restrict continuous models to copper-plate by default
        if commitment or args.all_fidelities_for_continuous:
            fidelities = [0, 1, 2]
        else:
            fidelities = [0]

        # AOS methods depend on commitment
        if commitment:
            aos_methods = ["binary", "gurobi_pool"]
        else:
            aos_methods = ["linear"]

        sizes = size_cache[commitment]

        print(f"\n===== commitment = {commitment}  fidelities = {fidelities} =====")
        for fidelity in fidelities:
            # ----------------------------------------------------------
            # Baseline extensive-form solve for this fidelity
            # ----------------------------------------------------------
            print(f"Baseline extensive  fidelity={fidelity} ...")
            base_row = run_baseline_extensive(
                case_path=str(case_path),
                fidelity=fidelity,
                commitment=commitment,
                epsilon=args.epsilon,
                time_limit=time_limit,
            )
            # Ensure size columns are always populated
            base_row["n_buses"] = sizes["n_buses"]
            base_row["n_lines"] = sizes["n_lines"]
            base_row["n_gens_or_commits"] = sizes["n_gens_or_commits"]
            rows.append(base_row)
            print(
                f"  -> {base_row['status']:8s}  "
                f"base={base_row['base_time_s']:6.1f}s  "
                f"obj={base_row['obj_value']}"
            )

            # ----------------------------------------------------------
            # Normal AOS matrix for this fidelity
            # ----------------------------------------------------------
            for aos_method in aos_methods:
                for rel_gap in rel_gaps:
                    # solve_mode innermost → extensive and benders are adjacent
                    for solve_mode in ["extensive", "benders"]:
                        key = (fidelity, commitment, solve_mode, aos_method)

                        # ---------- smart pruning ----------
                        if hit_cap.get(key, False):
                            print(
                                f"SKIPPED fid={fidelity} commit={commitment} "
                                f"mode={solve_mode} aos={aos_method} gap={rel_gap} "
                                f"(hit solution cap at earlier gap)"
                            )
                            skip_row = {
                                "case": case_path.name,
                                "grid_fidelity": fidelity,
                                "commitment": commitment,
                                "solve_mode": solve_mode,
                                "aos_method": aos_method,
                                "rel_gap": rel_gap,
                                "num_solutions_cap": num_solutions,
                                "time_limit_s": time_limit,
                                "base_time_s": float("nan"),
                                "aos_time_s": float("nan"),
                                "total_time_s": float("nan"),
                                "n_candidates": num_solutions,
                                "n_true": num_solutions,
                                "obj_value": float("nan"),
                                "master_lb_or_eta": float("nan"),
                                "n_cuts": float("nan"),
                                "n_iterations": float("nan"),
                                "n_buses": sizes["n_buses"],
                                "n_lines": sizes["n_lines"],
                                "n_gens_or_commits": sizes["n_gens_or_commits"],
                                "status": "skipped_due_to_earlier_cap",
                                "error_msg": "",
                                "solutions_summary": "",
                            }
                            rows.append(skip_row)
                            continue

                        # ---------- normal execution ----------
                        print(
                            f"Running fid={fidelity} commit={commitment} "
                            f"mode={solve_mode} aos={aos_method} gap={rel_gap} ..."
                        )
                        row = run_one_config(
                            case_path=str(case_path),
                            fidelity=fidelity,
                            commitment=commitment,
                            solve_mode=solve_mode,
                            aos_method=aos_method,
                            rel_gap=rel_gap,
                            num_solutions=num_solutions,
                            epsilon=args.epsilon,
                            time_limit=time_limit,
                            summary_threshold=summary_threshold,
                        )
                        # Ensure size columns are always populated
                        row["n_buses"] = sizes["n_buses"]
                        row["n_lines"] = sizes["n_lines"]
                        row["n_gens_or_commits"] = sizes["n_gens_or_commits"]

                        if (
                            not math.isnan(row["n_candidates"])
                            and row["n_candidates"] >= num_solutions
                        ):
                            # Mark that the discovery guarantee is broken
                            if row["status"] == "optimal":
                                row["status"] = "optimal_hit_cap"
                            hit_cap[key] = True
                            print(
                                f"  ** Hit solution cap ({num_solutions}). "
                                f"Status set to optimal_hit_cap. "
                                f"Larger gaps for this combination will be skipped."
                            )

                        rows.append(row)
                        print(
                            f"  -> {row['status']:8s}  "
                            f"base={row['base_time_s']:6.1f}s  "
                            f"aos={row['aos_time_s']:6.1f}s  "
                            f"cand={row['n_candidates']} true={row['n_true']}"
                        )

    # ---- Write CSV ----
    csv_path = out_dir / f"{case_path.stem}_results.csv"
    fieldnames = [
        "case",
        "n_buses",
        "n_lines",
        "n_gens_or_commits",
        "grid_fidelity",
        "commitment",
        "solve_mode",
        "aos_method",
        "rel_gap",
        "num_solutions_cap",
        "time_limit_s",
        "base_time_s",
        "aos_time_s",
        "total_time_s",
        "n_candidates",
        "n_true",
        "obj_value",
        "master_lb_or_eta",
        "n_cuts",
        "n_iterations",
        "status",
        "error_msg",
        "solutions_summary",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\\nWrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
