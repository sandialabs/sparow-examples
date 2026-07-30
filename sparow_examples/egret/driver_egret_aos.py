#!/usr/bin/env python3
"""
driver_egret_aos.py
===================

EGRET + SPAROW parallel of driver_single_pglib.py for AOS benchmarks.

Target
------
- Default: synthetic tiny UC (2-gen copperplate, T = n_periods) via the staged
  EGRET runners.
- Optional: real EGRET ModelData JSON via --egret_file (horizon truncated when
  --n-periods is also supplied; a WARNING is emitted).

Both extensive-form AOS and SPAROW classical Benders + AOS-Benders are
exercised with the Gurobi solution-pool method (and optionally binary
enumeration).

Ordering (mirrors the PGLIB driver)
-----------------------------------
  first_stage_mode → Baseline extensive → aos_method → rel_gap → solve_mode

so that:
  - a pure extensive-form baseline appears first for each first-stage mode,
  - extensive and Benders AOS runs for the same (mode, aos_method, gap) sit
    next to each other in the CSV.

Residual second-stage discretes / pure first-stage constraints on the
Benders path are handled exactly as in stage8_run_benders (the path that
matched EF objectives on EGRET UC):

  - Monkey-patch of BendersSolver._setup_topas_subproblem that forces
    remove_first_stage_only_cons=True and residual Binary/Integer → UnitInterval.
  - Do *not* pass subproblem_transforms=[relax_second_stage] to solve().
    That extra transform triggers SPAROW deepcopies that hit uncopyable
    EGRET fields (dict_keys / dangling _parent).

Master deliberately does NOT receive residual relaxation so first-stage
binaries stay discrete.

Assumptions
-----------
- Full stack: pyomo, egret, sparow, or_topas, gurobi
- Attachment / staged modules on PYTHONPATH (or next to this script)

Usage
-----
  # Synthetic (default)
  python driver_egret_aos.py --out_dir results/
  python driver_egret_aos.py --out_dir results/ --intensity 3 --n-periods 4
  python driver_egret_aos.py --out_dir results/ --intensity 2 --include-start-stop

  # Real EGRET JSON (horizon truncated to --n-periods with a warning)
  python driver_egret_aos.py --out_dir results/ --egret_file path/to/instance.json --n-periods 4

  # Explicit first-stage list
  python driver_egret_aos.py --out_dir results/ --egret_file path/to/instance.json \\
      --first-stage-names UnitOn,UnitStart,UnitStop
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyomo.environ as pyo

# ---------------------------------------------------------------------------
# Import path for attachment / local modules
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
logger = logging.getLogger("driver_egret_aos")

# Hard-coded solver names
BENDERS_SOLVER = "gurobi_persistent"
EF_SOLVER = "gurobi"
AOS_SOLVER = "gurobi"

INTENSITY_GAPS = {
    1: [0.0],
    2: [0.0, 0.01],
    3: [0.0, 0.01, 0.05],
    4: [0.0, 0.01, 0.05, 0.10],
}


def _print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Solution summary (same spirit as PGLIB simplify_solution)
# ---------------------------------------------------------------------------

def _sol_objective(sol) -> Optional[float]:
    try:
        if hasattr(sol, "objective") and callable(sol.objective):
            obj = sol.objective()
            return float(obj.value if hasattr(obj, "value") else obj)
        if hasattr(sol, "objective") and sol.objective is not None and not callable(
            sol.objective
        ):
            o = sol.objective
            return float(o.value if hasattr(o, "value") else o)
        if hasattr(sol, "objectives") and sol.objectives:
            o = sol.objectives[0]
            return float(o.value if hasattr(o, "value") else o)
    except Exception:
        pass
    return None


def _extract_uniton_compact(sol_or_model) -> str:
    """Compact UnitOn string for the solutions_summary column."""
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
                    pass
        elif hasattr(sol_or_model, "component_data_objects"):
            for v in sol_or_model.component_data_objects(
                pyo.Var, active=True, descend_into=True
            ):
                if "uniton" in v.name.lower():
                    try:
                        pairs.append((v.name, float(pyo.value(v))))
                    except Exception:
                        pass
    except Exception:
        return "extract_failed"

    if not pairs:
        return "no_UnitOn"
    # Sort by name for stable order; emit 0/1 string
    pairs.sort(key=lambda x: x[0])
    bits = "".join(
        "1" if (v == v and round(v) >= 0.5) else "0" for _, v in pairs
    )
    return f"UnitOn={bits}"


def _format_solution_summary(pool, threshold: int) -> str:
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
    try:
        iterable = list(pool)
    except TypeError:
        iterable = getattr(pool, "solutions", None) or getattr(pool, "_solutions", None) or []
        if hasattr(iterable, "values"):
            iterable = list(iterable.values())
        else:
            iterable = list(iterable) if iterable else []

    for sol in iterable:
        obj = _sol_objective(sol)
        unit = _extract_uniton_compact(sol)
        if obj is not None:
            parts.append(f"obj={obj:.6g};{unit}")
        else:
            parts.append(unit)
    return " || ".join(parts)


# ---------------------------------------------------------------------------
# Build EGRET SPAROW SP (synthetic or real JSON)
# ---------------------------------------------------------------------------

def build_egret_sp(
    n_periods: Optional[int] = 4,
    include_start_stop: bool = False,
    egret_file: Optional[str] = None,
    force_synthetic: bool = False,
    first_stage_names: Optional[Sequence[str]] = None,
) -> Tuple[Any, List[str]]:
    """
    Build a single-scenario SPAROW StochasticProgram from either the synthetic
    tiny UC or a real EGRET ModelData JSON.

    first_stage_names:
      - explicit list if supplied
      - UnitOn only (default heuristic)
      - UnitOn + UnitStart + UnitStop when include_start_stop=True

    When egret_file is given and n_periods is not None the horizon is truncated
    and an explicit WARNING is printed.
    """
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
            allow_synthetic=False,  # fail clearly if the file is bad
            allow_github_fetch=False,
            n_periods=n_periods,
        )

    binary_comps, _, suggested = stage2_inspect_variables(model)

    if first_stage_names is None:
        names = [b["name"] for b in binary_comps]
        if include_start_stop:
            first_stage_names = [
                n
                for n in names
                if any(k in n.lower() for k in ("uniton", "unitstart", "unitstop"))
            ] or suggested
        else:
            first_stage_names = [n for n in names if "uniton" in n.lower()] or suggested
    else:
        first_stage_names = list(first_stage_names)

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


def _case_label(
    egret_file: Optional[str],
    n_periods: Optional[int],
    force_synthetic: bool,
) -> str:
    """Stable case name for the CSV."""
    if force_synthetic or egret_file is None:
        T = n_periods if n_periods is not None else 4
        return f"tiny_uc_T{T}"
    stem = Path(egret_file).stem
    if n_periods is not None:
        return f"{stem}_T{n_periods}"
    return stem


# ---------------------------------------------------------------------------
# Base solves
# ---------------------------------------------------------------------------

def run_extensive_base(sp, time_limit: float) -> Dict[str, Any]:
    """Solve the extensive form; return model + objective + timing."""
    from sparow.ef import ExtensiveFormSolver

    t0 = time.perf_counter()
    ef_solver = ExtensiveFormSolver()
    ef_solver.set_options(
        solver=EF_SOLVER,
        solver_options={"TimeLimit": time_limit, "timelimit": time_limit},
    )
    status = "optimal"
    obj = float("nan")
    model = None
    try:
        res = ef_solver.solve_and_return_EF(sp)
        model = res.model
        # Prefer objective from results, fall back to model
        try:
            rd = res.to_dict() if hasattr(res, "to_dict") else None
            if rd and "solutions" in rd and rd["solutions"]:
                soln = next(iter(rd["solutions"].values()))
                if isinstance(soln, dict) and soln.get("objectives"):
                    obj = float(soln["objectives"][0]["value"])
        except Exception:
            pass
        if math.isnan(obj) and model is not None:
            objs = list(
                model.component_data_objects(pyo.Objective, active=True, descend_into=False)
            )
            if objs:
                obj = float(pyo.value(objs[0]))
    except Exception as exc:
        status = "error"
        return {
            "model": None,
            "base_time_s": time.perf_counter() - t0,
            "obj_value": float("nan"),
            "master_lb_or_eta": float("nan"),
            "n_cuts": 0,
            "n_iterations": 1,
            "status": status,
            "error_msg": f"{type(exc).__name__}: {str(exc)[:200]}",
        }

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
        "error_msg": "",
    }


def run_benders_base(
    sp,
    time_limit: float,
    max_iterations: int = 80,
    eta_lower: float = 0.0,
) -> Dict[str, Any]:
    """
    SPAROW BendersSolver aligned with the proven Stage-8 path for EGRET UC.

    Residual handling (exactly as stage8_run_benders):
      - Monkey-patch _setup_topas_subproblem to force
        remove_first_stage_only_cons=True and residual Binary/Integer → UnitInterval.
      - Do *not* pass subproblem_transforms=[relax_second_stage] to solve().
      - Accept additional_transforms (current or_topas API).

    Master capture for AOS-Benders is done *after* the dangerous Contingencies
    deepcopy, by intercepting BendersCutGenerator.add_subproblem and reading
    the generator's parent_block() (the upper_model).  We deliberately do NOT
    patch _transform_to_master_model: that mid-transform capture was turning
    the Contingencies uncopyable warning into a hard _parent crash.
    """
    from sparow.benders import BendersSolver

    t0 = time.perf_counter()
    eta_bounds_map = {b: (float(eta_lower), None) for b in sp.bundles}
    n_seeded, n_unfixed = _seed_and_unfix_first_stage(sp)
    _print(f"  first-stage: seeded={n_seeded}, unfixed={n_unfixed}")

    # Defensive: Contingencies (and similar EGRET Set initializers) are the
    # known uncopyable fields that turn copy.deepcopy(sp) into a recursive
    # _parent crash. Stage 8 / PoC survive because they run EF first on the
    # same SP; we still strip anything that still looks dangerous.
    n_stripped = 0
    try:
        for b_name in list(getattr(sp, "bundles", []) or []):
            try:
                model = sp.get_model(b_name) if hasattr(sp, "get_model") else None
            except Exception:
                model = None
            if model is None:
                # SPAROW SP often stores scenarios under .s[None, b]
                try:
                    model = sp.s[None, b_name]
                except Exception:
                    continue
            for name in ("Contingencies", "contingencies"):
                if hasattr(model, name):
                    try:
                        model.del_component(name)
                        n_stripped += 1
                    except Exception:
                        try:
                            setattr(model, name, None)
                            n_stripped += 1
                        except Exception:
                            pass
        if n_stripped:
            _print(f"  Stripped {n_stripped} Contingencies component(s) before Benders deepcopy")
    except Exception as e:
        _print(f"  (Contingencies strip skipped: {e})")

    captured = {"upper_model": None}

    # ------------------------------------------------------------------
    # 1. Capture master AFTER transform, via add_subproblem on the DATA class.
    #    or_topas uses @declare_custom_block so:
    #      BendersGenerator_Serial  = block class (no add_subproblem)
    #      Benders_Serial           = data class  (defines add_subproblem)
    #    parent_block() of the generator is the upper_model SPAROW just built.
    # ------------------------------------------------------------------
    _BCG = None
    _orig_add_subproblem = None
    _import_errors = []
    for _mod, _name in (
        ("or_topas.benders.benders_serial", "Benders_Serial"),
        ("or_topas.benders.benders_serial", "BendersGenerator_SerialData"),
        ("or_topas.benders", "Benders_Serial"),
        ("or_topas.benders.benders_serial", "BendersGenerator_Serial"),
        ("or_topas.benders", "BendersGenerator_Serial"),
    ):
        try:
            import importlib
            mod = importlib.import_module(_mod)
            cls = getattr(mod, _name)
        except Exception as e:
            _import_errors.append(f"{_mod}.{_name}: {e}")
            continue
        # Prefer a class that actually defines / inherits add_subproblem
        meth = getattr(cls, "add_subproblem", None)
        if meth is None:
            _import_errors.append(f"{_mod}.{_name}: no add_subproblem")
            continue
        _BCG = cls
        _orig_add_subproblem = meth
        _print(f"  Capture target: {_mod}.{_name}.add_subproblem")
        break
    if _BCG is None or _orig_add_subproblem is None:
        raise RuntimeError(
            "Could not locate or_topas Benders data class with add_subproblem. "
            "Tried: " + "; ".join(_import_errors)
        )

    def _capturing_add_subproblem(self, *args, **kwargs):
        try:
            parent = self.parent_block()
            if parent is not None:
                captured["upper_model"] = parent
        except Exception:
            pass
        return _orig_add_subproblem(self, *args, **kwargs)

    _BCG.add_subproblem = _capturing_add_subproblem

    # ------------------------------------------------------------------
    # 2. Stage-8 hardening of subproblem setup
    # ------------------------------------------------------------------
    _orig_setup = BendersSolver._setup_topas_subproblem

    def _setup_with_remove_fs_only_cons(
        sp_lower, b_lower, sp_upper, b_upper, remove_first_stage_objective_terms,
        additional_transforms=None,
        **kwargs,
    ):
        model_lower = BendersSolver._transform_to_subproblem_model(
            sp_lower,
            b_lower,
            default_domain=pyo.Reals,
            remove_first_stage_objective_terms=remove_first_stage_objective_terms,
            remove_first_stage_only_cons=True,  # critical for EGRET UC
        )
        n_relaxed = 0
        for v in model_lower.component_data_objects(pyo.Var, active=True, descend_into=True):
            if v.domain is pyo.Binary or (
                hasattr(v.domain, "name") and "binary" in str(v.domain).lower()
            ):
                v.domain = pyo.UnitInterval
                n_relaxed += 1
            elif str(v.domain).lower() in ("binary", "integers"):
                v.domain = pyo.UnitInterval
                n_relaxed += 1
        if n_relaxed:
            _print(
                f"  Subproblem setup: relaxed {n_relaxed} residual Binary/Integer "
                "vars to continuous [0,1] for dual cuts"
            )

        from pyomo.common.collections import ComponentMap

        complicating_variable_map = ComponentMap()
        for i, var_upper in sp_upper.int_to_FirstStageVar[b_upper].items():
            complicating_variable_map[var_upper] = sp_lower.int_to_FirstStageVar[
                b_lower
            ][i]
        return model_lower, complicating_variable_map

    BendersSolver._setup_topas_subproblem = staticmethod(_setup_with_remove_fs_only_cons)

    solver = BendersSolver()
    solver.set_options(
        solver=BENDERS_SOLVER,
        subproblem_solver=BENDERS_SOLVER,
        max_iterations=max_iterations,
        is_persistent_solver=True,
        allow_infeasible_subproblems=True,
        loglevel="WARNING",
    )
    for attr in ("allow_infeasible_subproblems", "allow_infeasible"):
        if hasattr(solver, attr):
            setattr(solver, attr, True)

    status = "optimal"
    obj = float("nan")
    n_iters = 0
    error_msg = ""
    try:
        _print(
            "  Calling BendersSolver.solve (Stage-8 style + late capture via "
            "BendersCutGenerator.add_subproblem) ..."
        )
        results = solver.solve(sp, eta_bounds_map)
        try:
            if hasattr(results, "to_dict"):
                d = results.to_dict()
                soln = next(iter(d["solutions"].values()))
                obj = float(soln["objectives"][0]["value"])
            if hasattr(results, "metadata"):
                md = results.metadata
                if hasattr(md, "get"):
                    n_iters = int(md.get("iterations", 0) or 0)
                elif hasattr(md, "iterations"):
                    n_iters = int(md.iterations)
        except Exception:
            pass

        upper = captured["upper_model"]
        if upper is not None and (math.isnan(obj) or obj is None):
            try:
                obj = float(pyo.value(upper.obj))
            except Exception:
                pass
    except Exception as exc:
        status = "error"
        error_msg = f"{type(exc).__name__}: {str(exc)[:300]}"
        import traceback

        _print("  *** Benders base solve traceback:")
        traceback.print_exc()
        upper = captured["upper_model"]
    finally:
        # Always restore both patches.
        BendersSolver._setup_topas_subproblem = _orig_setup
        _BCG.add_subproblem = _orig_add_subproblem

    elapsed = time.perf_counter() - t0
    if elapsed > time_limit + 1.0 and status == "optimal":
        status = "timeout"

    upper = captured["upper_model"]
    if upper is None and status != "error":
        status = "error"
        error_msg = (
            "failed to capture upper_model via BendersCutGenerator.add_subproblem "
            "(parent_block was None)"
        )

    # Post-capture sanity + structural diagnostics.
    if upper is not None and status != "error":
        benders_blocks = [
            b
            for b in upper.component_data_objects(pyo.Block, descend_into=True)
            if "BendersGenerator" in str(type(b))
        ]
        if not benders_blocks and not hasattr(upper, "benders"):
            status = "error"
            error_msg = (
                "Captured master has no BendersGenerator block – "
                "aos_benders_generate_candidates will refuse to run."
            )
            _print(f"  *** {error_msg}")
        else:
            n_vars = sum(
                1
                for _ in upper.component_data_objects(
                    pyo.Var, active=True, descend_into=True
                )
            )
            n_none_parent = 0
            for v in upper.component_data_objects(
                pyo.Var, active=True, descend_into=True
            ):
                try:
                    if v.parent_block() is None:
                        n_none_parent += 1
                except Exception:
                    n_none_parent += 1
            _print(
                f"  Captured master; BendersGenerator blocks={len(benders_blocks)}, "
                f"n_vars={n_vars}, n_none_parent={n_none_parent}, "
                f"has_benders={hasattr(upper, 'benders')}"
            )
            if n_none_parent > 0:
                _print(
                    f"  WARNING: {n_none_parent} vars have parent_block() is None – "
                    "AOS-Benders may still fail with _parent errors"
                )

    return {
        "model": upper,
        "base_time_s": elapsed,
        "obj_value": obj,
        "master_lb_or_eta": obj,
        "n_cuts": float("nan"),
        "n_iterations": n_iters if n_iters else float("nan"),
        "status": status,
        "error_msg": error_msg,
    }


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

    n = len(pool)
    return {
        "aos_time_s": elapsed,
        "n_candidates": n,
        "n_true": n,  # EF pool is already the true set
        "obj_value": float("nan"),
        "status": status,
        "error_msg": "",
        "pool": pool if keep_pool else None,
    }


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
        "gurobi_pool": "gurobi_pool",
        "binary": "binary",
        "linear": "linear",
    }
    enumeration_method = enum_map.get(aos_method)
    if enumeration_method is None:
        return {
            "aos_time_s": time.perf_counter() - t0,
            "n_candidates": float("nan"),
            "n_true": float("nan"),
            "obj_value": float("nan"),
            "status": "error",
            "error_msg": f"Unknown aos_method for benders: {aos_method}",
            "pool": None,
        }

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
            "error_msg": "",
            "pool": None,
        }

    true_pool = aos_benders_filter(candidate_pool, data, tee=False, tee_final=False)
    elapsed = time.perf_counter() - t0
    status = "timeout" if elapsed > time_limit else "optimal"
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
        "status": status,
        "error_msg": "",
        "pool": true_pool if keep_pool else None,
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
    # Optional precomputed base solves (avoids repeated SPAROW deepcopy).
    cached_ef_base: Optional[Dict[str, Any]] = None,
    cached_benders_base: Optional[Dict[str, Any]] = None,
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
        # ---- base solve (use cache when provided) ----
        if solve_mode == "extensive":
            if cached_ef_base is not None:
                base = cached_ef_base
            else:
                sp, _ = build_egret_sp(
                    n_periods=n_periods,
                    include_start_stop=include_start_stop,
                    egret_file=egret_file,
                    force_synthetic=force_synthetic,
                    first_stage_names=first_stage_names,
                )
                row["n_first_stage"] = len(
                    getattr(sp, "int_to_FirstStageVar", {}).get(
                        next(iter(sp.bundles)), {}
                    )
                )
                base = run_extensive_base(sp, time_limit=time_limit)
        else:
            if cached_benders_base is not None:
                base = cached_benders_base
            else:
                sp, _ = build_egret_sp(
                    n_periods=n_periods,
                    include_start_stop=include_start_stop,
                    egret_file=egret_file,
                    force_synthetic=force_synthetic,
                    first_stage_names=first_stage_names,
                )
                row["n_first_stage"] = len(
                    getattr(sp, "int_to_FirstStageVar", {}).get(
                        next(iter(sp.bundles)), {}
                    )
                )
                base = run_benders_base(
                    sp,
                    time_limit=time_limit,
                    max_iterations=max_benders_iterations,
                )

        if math.isnan(row["n_first_stage"]) and base.get("n_first_stage") is not None:
            row["n_first_stage"] = base["n_first_stage"]

        row["base_time_s"] = base["base_time_s"]
        row["n_cuts"] = base.get("n_cuts", 0)
        row["n_iterations"] = base.get("n_iterations", 1)
        row["master_lb_or_eta"] = base["master_lb_or_eta"]
        row["status"] = base["status"]
        row["obj_value"] = base.get("obj_value", base["master_lb_or_eta"])
        if base.get("error_msg"):
            row["error_msg"] = base["error_msg"]
        if base.get("n_first_stage") is not None and math.isnan(row["n_first_stage"]):
            row["n_first_stage"] = base["n_first_stage"]

        # ---- AOS phase ----
        if base["status"] not in ("error",) and base.get("model") is not None:
            if solve_mode == "extensive":
                aos_res = run_aos_extensive(
                    model=base["model"],
                    aos_method=aos_method,
                    rel_gap=rel_gap,
                    num_solutions=num_solutions,
                    time_limit=time_limit,
                    keep_pool=True,
                )
            else:
                aos_res = run_aos_benders(
                    m=base["model"],
                    aos_method=aos_method,
                    rel_gap=rel_gap,
                    num_solutions=num_solutions,
                    time_limit=time_limit,
                    keep_pool=True,
                )

            row["aos_time_s"] = aos_res["aos_time_s"]
            row["n_candidates"] = aos_res["n_candidates"]
            row["n_true"] = aos_res["n_true"]
            if aos_res["status"] in ("timeout", "error"):
                row["status"] = aos_res["status"]
            if aos_res.get("error_msg"):
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
        import traceback

        traceback.print_exc()

    return row


def run_baseline_extensive(
    *,
    n_periods: Optional[int],
    include_start_stop: bool,
    time_limit: float,
    case: str,
    egret_file: Optional[str] = None,
    force_synthetic: bool = False,
    first_stage_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Pure extensive-form solve used as an accuracy reference."""
    fs_label = "UnitOn+StartStop" if include_start_stop else "UnitOn"
    row: Dict[str, Any] = {
        "case": case,
        "n_periods": n_periods if n_periods is not None else float("nan"),
        "first_stage_mode": f"{fs_label}/Baseline",
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
        "n_first_stage": float("nan"),
        "status": "error",
        "error_msg": "",
        "solutions_summary": "",
    }
    try:
        sp, fs_names = build_egret_sp(
            n_periods=n_periods,
            include_start_stop=include_start_stop,
            egret_file=egret_file,
            force_synthetic=force_synthetic,
            first_stage_names=first_stage_names,
        )
        row["n_first_stage"] = len(
            getattr(sp, "int_to_FirstStageVar", {}).get(next(iter(sp.bundles)), {})
        )
        base = run_extensive_base(sp, time_limit=time_limit)
        row["base_time_s"] = base["base_time_s"]
        row["total_time_s"] = base["base_time_s"]
        row["obj_value"] = base["obj_value"]
        row["master_lb_or_eta"] = base["master_lb_or_eta"]
        row["status"] = base["status"]
        row["error_msg"] = base.get("error_msg", "")
        row["solutions_summary"] = f"obj={base['obj_value']}"
    except Exception as exc:
        row["status"] = "error"
        row["error_msg"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "EGRET + SPAROW AOS driver (parallel of driver_single_pglib.py) – "
            "synthetic tiny UC or real EGRET JSON, EF AOS + SPAROW Benders AOS-Benders"
        )
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory for the results CSV",
    )
    parser.add_argument(
        "--egret_file",
        type=str,
        default=None,
        help=(
            "Path to EGRET ModelData JSON. If omitted (or with --force-synthetic) "
            "the synthetic 2-gen copperplate UC is used."
        ),
    )
    parser.add_argument(
        "--force-synthetic",
        action="store_true",
        default=False,
        help="Ignore --egret_file and always build the synthetic tiny UC.",
    )
    parser.add_argument(
        "--n-periods",
        type=int,
        default=4,
        help=(
            "Number of time periods (sizes the synthetic instance, or truncates "
            "a real JSON horizon; a WARNING is emitted on truncation)."
        ),
    )
    parser.add_argument(
        "--include-start-stop",
        action="store_true",
        default=False,
        help=(
            "Also run a matrix with UnitStart/UnitStop in the first-stage list "
            "(default: UnitOn only). Ignored when --first-stage-names is given."
        ),
    )
    parser.add_argument(
        "--first-stage-names",
        type=str,
        default=None,
        help=(
            "Comma-separated first-stage component names "
            "(default: heuristic UnitOn, or UnitOn+StartStop with --include-start-stop)."
        ),
    )
    parser.add_argument(
        "--max_sols",
        type=int,
        default=50,
        help="num_solutions cap for the pool / AOS methods",
    )
    parser.add_argument(
        "--time_limit",
        type=float,
        default=120.0,
        help="Wall-clock limit (seconds) per base / AOS phase",
    )
    parser.add_argument(
        "--max-benders-iter",
        type=int,
        default=80,
        help="Max SPAROW Benders iterations",
    )
    parser.add_argument(
        "--intensity",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help=(
            "Relative-gap levels: "
            "1=[0.0], 2=[0.0,0.01], 3=[0.0,0.01,0.05], 4=[0.0,0.01,0.05,0.10]"
        ),
    )
    parser.add_argument(
        "--summary_threshold",
        type=int,
        default=5,
        help=(
            "When n_true ≤ this value, write a compact UnitOn summary into "
            "solutions_summary (default 5)"
        ),
    )
    parser.add_argument(
        "--aos-methods",
        nargs="+",
        default=["gurobi_pool"],
        choices=["gurobi_pool", "binary"],
        help="AOS enumeration methods to run (default: gurobi_pool only)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse optional explicit first-stage list
    fs_override: Optional[List[str]] = None
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
    _print(f"egret_file = {args.egret_file}")
    _print(f"force_synthetic = {args.force_synthetic}")
    _print(
        f"first_stage_modes = "
        f"{[('UnitOn+StartStop' if m else 'UnitOn') for m in first_stage_modes]}"
    )
    if fs_override is not None:
        _print(f"first_stage_names override = {fs_override}")
    _print(f"aos_methods = {args.aos_methods}")
    _print(f"summary_threshold = {args.summary_threshold}")
    _print("Order: first_stage_mode → Baseline → aos_method → rel_gap → solve_mode")

    rows: List[Dict[str, Any]] = []
    hit_cap: Dict[Tuple, bool] = {}

    for include_ss in first_stage_modes:
        fs_label = "UnitOn+StartStop" if include_ss else "UnitOn"
        _print(f"\n===== first_stage_mode = {fs_label} =====")

        # ------------------------------------------------------------------
        # Run each base solve ONCE, then only vary AOS gap/method.
        # Benders is deliberately run before any pool AOS so SPAROW's
        # deepcopy of the SP happens on a clean process state.  The captured
        # master is reused for every Benders AOS gap (no rebuild).
        # ------------------------------------------------------------------
        _print(f"Building SP + baseline extensive  mode={fs_label} ...")
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

        base_row = run_baseline_extensive(
            n_periods=args.n_periods,
            include_start_stop=include_ss,
            time_limit=args.time_limit,
            case=case,
            egret_file=args.egret_file,
            force_synthetic=args.force_synthetic,
            first_stage_names=fs_override,
        )
        base_row["n_first_stage"] = n_fs
        rows.append(base_row)
        _print(
            f"  -> {base_row['status']:8s}  "
            f"base={base_row['base_time_s']:6.1f}s  "
            f"obj={base_row['obj_value']}"
        )

        _print(f"EF base solve (once, reused for all extensive AOS gaps) ...")
        sp_ef, _ = build_egret_sp(
            n_periods=args.n_periods,
            include_start_stop=include_ss,
            egret_file=args.egret_file,
            force_synthetic=args.force_synthetic,
            first_stage_names=fs_override,
        )
        cached_ef = run_extensive_base(sp_ef, time_limit=args.time_limit)
        cached_ef["n_first_stage"] = n_fs
        _print(
            f"  -> {cached_ef['status']:8s}  "
            f"base={cached_ef['base_time_s']:6.1f}s  "
            f"obj={cached_ef['obj_value']}"
        )

        _print(f"Benders base solve (once, reused for all Benders AOS gaps) ...")
        # CRITICAL: reuse the same SP that already went through EF.
        # Stage 8 and the working PoC both do EF then Benders on one SP.
        # A fresh SP still has Contingencies in an uncopyable state; after EF
        # the deepcopy inside SPAROW _create_sp_upper succeeds (with a warning).
        _print("  Reusing EF SP for Benders (same path as Stage 8 / PoC)")
        cached_benders = run_benders_base(
            sp_ef,
            time_limit=args.time_limit,
            max_iterations=args.max_benders_iter,
        )
        cached_benders["n_first_stage"] = n_fs
        _print(
            f"  -> {cached_benders['status']:8s}  "
            f"base={cached_benders['base_time_s']:6.1f}s  "
            f"obj={cached_benders['obj_value']}"
        )
        if cached_benders["status"] == "error":
            _print(f"  *** Benders base error: {cached_benders.get('error_msg')}")

        for aos_method in args.aos_methods:
            for rel_gap in rel_gaps:
                for solve_mode in ["extensive", "benders"]:
                    key = (include_ss, solve_mode, aos_method)

                    if hit_cap.get(key, False):
                        _print(
                            f"SKIPPED mode={fs_label} solve={solve_mode} "
                            f"aos={aos_method} gap={rel_gap} "
                            f"(hit solution cap at earlier gap)"
                        )
                        skip_row = {
                            "case": case,
                            "n_periods": args.n_periods,
                            "first_stage_mode": fs_label,
                            "solve_mode": solve_mode,
                            "aos_method": aos_method,
                            "rel_gap": rel_gap,
                            "num_solutions_cap": args.max_sols,
                            "time_limit_s": args.time_limit,
                            "base_time_s": float("nan"),
                            "aos_time_s": float("nan"),
                            "total_time_s": float("nan"),
                            "n_candidates": args.max_sols,
                            "n_true": args.max_sols,
                            "obj_value": float("nan"),
                            "master_lb_or_eta": float("nan"),
                            "n_cuts": float("nan"),
                            "n_iterations": float("nan"),
                            "n_first_stage": n_fs,
                            "status": "skipped_due_to_earlier_cap",
                            "error_msg": "",
                            "solutions_summary": "",
                        }
                        rows.append(skip_row)
                        continue

                    _print(
                        f"Running mode={fs_label} solve={solve_mode} "
                        f"aos={aos_method} gap={rel_gap} ..."
                    )
                    row = run_one_config(
                        n_periods=args.n_periods,
                        include_start_stop=include_ss,
                        solve_mode=solve_mode,
                        aos_method=aos_method,
                        rel_gap=rel_gap,
                        num_solutions=args.max_sols,
                        time_limit=args.time_limit,
                        max_benders_iterations=args.max_benders_iter,
                        summary_threshold=args.summary_threshold,
                        case=case,
                        egret_file=args.egret_file,
                        force_synthetic=args.force_synthetic,
                        first_stage_names=fs_override,
                        cached_ef_base=cached_ef if solve_mode == "extensive" else None,
                        cached_benders_base=(
                            cached_benders if solve_mode == "benders" else None
                        ),
                    )
                    row["n_first_stage"] = n_fs

                    if (
                        not math.isnan(row["n_candidates"])
                        and row["n_candidates"] >= args.max_sols
                    ):
                        if row["status"] == "optimal":
                            row["status"] = "optimal_hit_cap"
                        hit_cap[key] = True
                        _print(
                            f"  ** Hit solution cap ({args.max_sols}). "
                            f"Larger gaps for this combination will be skipped."
                        )

                    rows.append(row)
                    msg = (
                        f"  -> {row['status']:8s}  "
                        f"base={row['base_time_s']:6.1f}s  "
                        f"aos={row['aos_time_s']:6.1f}s  "
                        f"cand={row['n_candidates']} true={row['n_true']}"
                    )
                    if row["status"] == "error" and row.get("error_msg"):
                        msg += f"\n     error: {row['error_msg']}"
                    _print(msg)

    # ---- Write CSV ----
    csv_path = out_dir / f"{case}_results.csv"
    fieldnames = [
        "case",
        "n_periods",
        "first_stage_mode",
        "n_first_stage",
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

    _print(f"\nWrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
