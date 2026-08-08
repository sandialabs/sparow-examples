#!/usr/bin/env python3
"""
aos_gtep_runner.py
==================

General runner for SPAROW + GTEP stochastic programs + OR-TOPAS alternative
solutions.

Designed as the GTEP counterpart of the thin aos_single_ef.py driver while
accommodating the code-generation packaging used for GTEP models:

  - Accepts any module that exports create_sp() (output of setup_load_scens.py /
    setup_multiple_load_scens.py, or a hand-written equivalent).
  - Supports both extensive-form (EF) and classical Benders solve paths.
  - Defaults to a first-stage-only solution archive (~25 KB) because full EF
    solutions for even the 9-bus model are ~50 MB each.  Full archive is
    available behind a flag.
  - No relative-gap parametric matrix (capacity-expansion models are
    comparatively degenerate).

Typical usage
-------------
  # EF path, first-stage archive only (default)
  python aos_gtep_runner.py \\
      --sp-module sparow_examples.aos_gtep_9bus.aos_single

  # Multi-scenario package produced by setup_multiple_load_scens.py
  python aos_gtep_runner.py \\
      --sp-module sparow_examples.aos_gtep_9bus.benders_multi \\
      --mode ef --aos-method gurobi_pool --num-solutions 5

  # Benders path (uses aos-benders generate + filter)
  python aos_gtep_runner.py \\
      --sp-module sparow_examples.aos_gtep_9bus.aos_single \\
      --mode benders --num-solutions 3

  # Hamming binary enumeration + full archive (debug, EF)
  python aos_gtep_runner.py \\
      --sp-module sparow_examples.aos_gtep_9bus.aos_single \\
      --aos-method hamming --full-archive

Residual second-stage discretes are handled exactly as in the staged GTEP
runners and aos_single_ef.py:

  EF path:     sp.add_transformation(relax_second_stage)
  Benders path: BendersSolver.solve(..., subproblem_transforms=[relax_second_stage],
                                          master_transforms=None)
                so the master keeps first-stage binaries discrete.

IMPORTANT – Benders AOS:
  Ordinary AOS kernels on the master only produce *candidates*.  True
  alternative solutions require the OR-TOPAS aos-benders pipeline
  (aos_benders_generate_candidates + aos_benders_filter) that re-evaluates
  the subproblems and retains only solutions that remain feasible and
  within the optimality gap.  This runner uses that pipeline on the
  Benders path.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import pyomo.environ as pyo


# ---------------------------------------------------------------------------
# create_sp resolution
# ---------------------------------------------------------------------------

def load_create_sp(module_path: str) -> Callable[[], Any]:
    """Import module_path and return its create_sp callable."""
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        raise SystemExit(
            f"Could not import create_sp source '{module_path}': {exc}\n"
            "Ensure the generated package (output of setup_load_scens.py / "
            "setup_multiple_load_scens.py) is on PYTHONPATH."
        ) from exc
    fn = getattr(mod, "create_sp", None)
    if fn is None or not callable(fn):
        raise SystemExit(
            f"Module '{module_path}' does not export a callable create_sp()."
        )
    return fn


# ---------------------------------------------------------------------------
# First-stage name set and projection (from aos_single_scenario_ef_hamming_var_split)
# ---------------------------------------------------------------------------

def extract_first_stage_name_set(sp) -> Set[str]:
    """
    Collect the authoritative first-stage name set from the live SPAROW maps
    after the EF (or master) has been built.  Falls back to a name-pattern
    heuristic only if the maps are empty.
    """
    fs_name_set: Set[str] = set()
    try:
        b0 = next(iter(sp.bundles))
        if hasattr(sp, "int_to_FirstStageVar") and b0 in sp.int_to_FirstStageVar:
            fs_name_set = {v.name for v in sp.int_to_FirstStageVar[b0].values()}
        elif hasattr(sp, "int_to_FirstStageVarName") and sp.int_to_FirstStageVarName:
            fs_name_set = set(sp.int_to_FirstStageVarName.values())
    except Exception:
        pass
    return fs_name_set


def make_is_first_stage(fs_name_set: Set[str]):
    """Return a predicate that recognises first-stage variable names."""
    if fs_name_set:
        def is_first_stage(name: str) -> bool:
            return name in fs_name_set
        return is_first_stage

    print(
        "  WARNING: SPAROW first-stage maps empty; falling back to name patterns",
        flush=True,
    )
    def is_first_stage(name: str) -> bool:
        n = (name or "").lower()
        return ("investmentstage" in n) and (
            "binary_indicator" in n or "renewable" in n
        )
    return is_first_stage


def project_to_first_stage(sol, is_first_stage) -> Dict[str, Any]:
    """
    Project a Solution onto the first-stage variables.

    Returns a dict compatible with
    sparow.sp.util.constrain_EF_model(..., first_stage_variables=...).
    """
    fs = {}
    # Support both OR-TOPAS Solution objects and simple dicts / pool entries
    if hasattr(sol, "variables") and callable(sol.variables):
        for vinfo in sol.variables():
            if vinfo.name and is_first_stage(vinfo.name):
                fs[vinfo.name] = float(vinfo.value) if vinfo.value is not None else None
        objs = list(sol.objectives()) if hasattr(sol, "objectives") else []
        obj_val = float(objs[0].value) if objs else None
    elif isinstance(sol, dict):
        # pool entry style
        vars_ = sol.get("variables") or sol.get("first_stage_variables") or {}
        if isinstance(vars_, dict):
            for name, val in vars_.items():
                if is_first_stage(name):
                    fs[name] = float(val) if val is not None else None
        obj_val = sol.get("objective")
    else:
        obj_val = None
        # best-effort attribute walk
        for attr in ("_variables", "variables"):
            vars_list = getattr(sol, attr, None)
            if vars_list is not None:
                for v in vars_list:
                    name = getattr(v, "name", None) or str(v)
                    if is_first_stage(name):
                        val = getattr(v, "value", None)
                        fs[name] = float(val) if val is not None else None
                break
    return {
        "id": getattr(sol, "id", None),
        "objective": obj_val,
        "first_stage_variables": fs,
    }


# ---------------------------------------------------------------------------
# AOS kernels (EF path – ordinary AOS is correct on the full EF)
# ---------------------------------------------------------------------------

def run_aos_on_model(
    model,
    *,
    aos_method: str,
    num_solutions: int,
    rel_opt_gap: float,
    tee: bool = False,
):
    """
    Dispatch to the chosen OR-TOPAS AOS kernel (for the *EF* path only).

    aos_method:
      "gurobi_pool" – or_topas.aos.gurobi_generate_solutions (PoolSearchMode=2)
      "binary"      – or_topas.aos.enumerate_binary_solutions (Balas no-good)
      "hamming"     – enumerate_binary_solutions with search_mode="hamming"
    """
    import or_topas.aos

    if aos_method == "gurobi_pool":
        return or_topas.aos.gurobi_generate_solutions(
            model,
            num_solutions=num_solutions,
            rel_opt_gap=rel_opt_gap,
            pool_search_mode=2,
            tee=tee,
        )
    if aos_method in ("binary", "hamming"):
        kwargs = dict(
            model=model,
            num_solutions=num_solutions,
            rel_opt_gap=rel_opt_gap,
            solver="gurobi",
            tee=tee,
        )
        if aos_method == "hamming":
            kwargs["search_mode"] = "hamming"
        return or_topas.aos.enumerate_binary_solutions(**kwargs)
    raise ValueError(f"Unknown aos_method: {aos_method!r}")


# ---------------------------------------------------------------------------
# EF path
# ---------------------------------------------------------------------------

def run_ef_path(
    sp,
    *,
    aos_method: str,
    num_solutions: int,
    rel_opt_gap: float,
    tee: bool = False,
):
    """Build compact EF, run ordinary AOS, return (aos_results, model, fs_name_set)."""
    from sparow.sp.util import relax_second_stage

    print("\n--- EF path: registering relax_second_stage ---", flush=True)
    sp.add_transformation(relax_second_stage)

    print("--- Creating compact extensive form ---", flush=True)
    model = sp.create_EF(compact_repn=True)

    fs_name_set = extract_first_stage_name_set(sp)
    print(f"  First-stage name set size: {len(fs_name_set)}", flush=True)
    if fs_name_set:
        for n in sorted(fs_name_set)[:6]:
            print(f"    {n}", flush=True)
        if len(fs_name_set) > 6:
            print(f"    ... +{len(fs_name_set) - 6} more", flush=True)

    print(f"\n--- Running AOS ({aos_method}) on EF ---", flush=True)
    t0 = time.perf_counter()
    aos_results = run_aos_on_model(
        model,
        aos_method=aos_method,
        num_solutions=num_solutions,
        rel_opt_gap=rel_opt_gap,
        tee=tee,
    )
    elapsed = time.perf_counter() - t0
    print(f"  AOS finished in {elapsed:.2f} s, {len(aos_results)} solution(s)", flush=True)
    for i, s in enumerate(aos_results):
        try:
            print(f"  Sol {i}: objective = {s.objective().value}", flush=True)
        except Exception:
            print(f"  Sol {i}: (could not read objective)", flush=True)

    return aos_results, model, fs_name_set


# ---------------------------------------------------------------------------
# Benders path – MUST use aos-benders generate + filter for true solutions
# ---------------------------------------------------------------------------

def _seed_and_unfix_first_stage(sp) -> tuple[int, int]:
    """Unfix + seed first-stage variables fixed from in_service data."""
    n_seeded = n_unfixed = 0
    fs_map_all = getattr(sp, "int_to_FirstStageVar", {}) or {}
    for b in getattr(sp, "bundles", []):
        for v in fs_map_all.get(b, {}).values():
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
                if v.lb is not None:
                    v.set_value(v.lb, skip_validation=True)
                elif v.ub is not None:
                    v.set_value(v.ub, skip_validation=True)
                else:
                    v.set_value(0.0, skip_validation=True)
                n_seeded += 1
    return n_seeded, n_unfixed


def _import_aos_benders():
    """Prefer local aos_benders.py, fall back to or_topas.benders.aos_benders."""
    try:
        from aos_benders import (
            aos_benders_generate_candidates,
            aos_benders_filter,
        )
        return aos_benders_generate_candidates, aos_benders_filter
    except ImportError:
        from or_topas.benders.aos_benders import (
            aos_benders_generate_candidates,
            aos_benders_filter,
        )
        return aos_benders_generate_candidates, aos_benders_filter


def _capture_master_via_add_subproblem():
    """
    EGRET-style capture of the upper_model / master by intercepting
    Benders_Serial.add_subproblem (or equivalent).  Returns a dict that will
    be populated with the captured master when the solver runs.
    """
    captured = {"upper_model": None}
    _BCG = None
    _orig = None
    import_errors = []
    for mod_name, cls_name in (
        ("or_topas.benders.benders_serial", "Benders_Serial"),
        ("or_topas.benders.benders_serial", "BendersGenerator_SerialData"),
        ("or_topas.benders", "Benders_Serial"),
        ("or_topas.benders.benders_serial", "BendersGenerator_Serial"),
        ("or_topas.benders", "BendersGenerator_Serial"),
    ):
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            meth = getattr(cls, "add_subproblem", None)
            if meth is None:
                import_errors.append(f"{mod_name}.{cls_name}: no add_subproblem")
                continue
            _BCG = cls
            _orig = meth
            break
        except Exception as e:
            import_errors.append(f"{mod_name}.{cls_name}: {e}")
            continue

    if _BCG is None or _orig is None:
        return captured, None, None, import_errors

    def _capturing_add_subproblem(self, *args, **kwargs):
        try:
            parent = self.parent_block()
            if parent is not None:
                captured["upper_model"] = parent
        except Exception:
            pass
        return _orig(self, *args, **kwargs)

    _BCG.add_subproblem = _capturing_add_subproblem
    return captured, _BCG, _orig, []


def run_benders_path(
    sp,
    *,
    aos_method: str,
    num_solutions: int,
    rel_opt_gap: float,
    max_iterations: int = 50,
    tee: bool = False,
):
    """
    Classical Benders via SPAROW BendersSolver, then **aos-benders**
    (generate candidates + filter) so only true solutions are retained.

    Residual second-stage discretes are relaxed only on the subproblems
    (subproblem_transforms=[relax_second_stage]); the master keeps first-stage
    binaries discrete — identical contract to stage8 of gtep_sparow_staged.py.
    """
    from sparow.benders import BendersSolver
    from sparow.sp.util import relax_second_stage

    print("\n--- Benders path ---", flush=True)
    print(
        "  subproblem_transforms=[relax_second_stage], master_transforms=None",
        flush=True,
    )
    print(
        "  AOS will use aos_benders_generate_candidates + aos_benders_filter "
        "(candidates alone are NOT true solutions)",
        flush=True,
    )

    try:
        sp.add_transformation(relax_second_stage)
    except Exception:
        pass

    n_seeded, n_unfixed = _seed_and_unfix_first_stage(sp)
    print(f"  first-stage: unfixed={n_unfixed}, seeded={n_seeded}", flush=True)

    # Capture master via add_subproblem monkey-patch (EGRET pattern)
    captured, BCG, orig_add, capture_errs = _capture_master_via_add_subproblem()
    if BCG is not None:
        print(f"  Master capture: patched {BCG.__name__}.add_subproblem", flush=True)
    else:
        print(
            "  WARNING: could not install add_subproblem capture "
            f"({'; '.join(capture_errs[:3])}). Will try attribute fallbacks.",
            flush=True,
        )

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

    # SPAROW asserts len(eta_bounds_map) > 0 inside _transform_to_master_model.
    # Staged Stage 8 uses the same pattern; a modest negative lower bound is a
    # safe baseline when the true subproblem objective can be positive.
    eta_bounds_map = {b: (-1e6, None) for b in sp.bundles}
    if not eta_bounds_map:
        raise RuntimeError(
            "No bundles on SP; cannot build eta_bounds_map for Benders master."
        )
    print(f"  eta_bounds_map: {eta_bounds_map}", flush=True)

    print(
        f"  Calling BendersSolver.solve (max_iterations={max_iterations}) ...",
        flush=True,
    )
    t0 = time.perf_counter()
    try:
        results = solver.solve(
            sp,
            eta_bounds_map,
            subproblem_transforms=[relax_second_stage],
            master_transforms=None,
        )
    finally:
        # Always restore the patch
        if BCG is not None and orig_add is not None:
            BCG.add_subproblem = orig_add
    elapsed = time.perf_counter() - t0
    print(f"  Benders finished in {elapsed:.2f} s", flush=True)

    # Resolve master model
    master = captured.get("upper_model")
    if master is None:
        for attr in ("master", "master_model", "_master", "model", "upper_model"):
            candidate = getattr(solver, attr, None) or getattr(results, attr, None)
            if candidate is not None and hasattr(candidate, "component_objects"):
                master = candidate
                print(f"  Master recovered via attribute '{attr}'", flush=True)
                break

    if master is None:
        raise RuntimeError(
            "Could not recover a master model after Benders solve.  "
            "AOS-Benders requires a model that contains a BendersGenerator block. "
            "Check SPAROW / OR-TOPAS version or extend the capture logic."
        )

    # Sanity: must have a BendersGenerator block for aos_benders
    benders_blocks = [
        b
        for b in master.component_data_objects(pyo.Block, descend_into=True)
        if "BendersGenerator" in str(type(b))
    ]
    if not benders_blocks and not hasattr(master, "benders"):
        raise RuntimeError(
            "Recovered master has no BendersGenerator block; "
            "aos_benders_generate_candidates will refuse to run. "
            "Master capture may have grabbed the wrong object."
        )
    print(
        f"  Master ready; BendersGenerator blocks = {len(benders_blocks)}",
        flush=True,
    )

    fs_name_set = extract_first_stage_name_set(sp)
    print(f"  First-stage name set size: {len(fs_name_set)}", flush=True)

    # ---- aos-benders: generate candidates then filter ----
    enum_map = {
        "gurobi_pool": "gurobi_pool",
        "binary": "binary",
        "hamming": "binary",  # Hamming is a binary search mode; map to binary for aos-benders
    }
    enumeration_method = enum_map.get(aos_method)
    if enumeration_method is None:
        raise ValueError(
            f"aos_method={aos_method!r} is not supported on the Benders path. "
            "Use gurobi_pool, binary, or hamming."
        )

    aos_benders_generate_candidates, aos_benders_filter = _import_aos_benders()

    print(
        f"\n--- aos-benders: generate_candidates (enumeration_method={enumeration_method}) ---",
        flush=True,
    )
    t0 = time.perf_counter()
    candidate_pool, data = aos_benders_generate_candidates(
        m=master,
        rel_gap=rel_opt_gap,
        num_solutions=num_solutions,
        mip_solver="gurobi",
        enumeration_method=enumeration_method,
        tee=tee,
    )
    print(
        f"  Candidates: {len(candidate_pool)}  ({time.perf_counter() - t0:.2f} s)",
        flush=True,
    )

    print("--- aos-benders: filter (true solutions only) ---", flush=True)
    t0 = time.perf_counter()
    true_pool = aos_benders_filter(
        candidate_pool, data, tee=tee, tee_final=False
    )
    print(
        f"  True solutions retained: {len(true_pool)}  ({time.perf_counter() - t0:.2f} s)",
        flush=True,
    )

    return true_pool, master, fs_name_set


# ---------------------------------------------------------------------------
# Archive writing
# ---------------------------------------------------------------------------

def write_archives(
    aos_results,
    fs_name_set: Set[str],
    *,
    out_prefix: str,
    full_archive: bool,
    metadata: Dict[str, Any],
):
    """Write first-stage archive (always) and optionally the full archive."""
    is_first_stage = make_is_first_stage(fs_name_set)

    solutions = []
    try:
        iterable = list(aos_results)
    except TypeError:
        iterable = []
        if hasattr(aos_results, "solutions"):
            try:
                iterable = list(aos_results.solutions.values())
            except Exception:
                pass
        if not iterable and hasattr(aos_results, "to_dict"):
            try:
                d = aos_results.to_dict()
                iterable = list(d.get("solutions", {}).values())
            except Exception:
                pass
        # PyomoPoolManager style
        if not iterable and hasattr(aos_results, "get_pool"):
            try:
                pool = aos_results.get_pool("aos_benders") or aos_results.get_pool()
                iterable = list(pool) if pool is not None else []
            except Exception:
                pass

    for s in iterable:
        try:
            solutions.append(project_to_first_stage(s, is_first_stage))
        except Exception as exc:
            print(f"  WARNING: could not project a solution ({exc})", flush=True)

    fs_archive = {
        "metadata": {
            **metadata,
            "description": (
                "First-stage projection of AOS solutions "
                "(investment indicators only).  On the Benders path these are "
                "true solutions after aos_benders_filter."
            ),
            "num_solutions": len(solutions),
            "first_stage_name_set_size": len(fs_name_set),
        },
        "solutions": solutions,
    }
    if solutions:
        n_fs = len(solutions[0].get("first_stage_variables", {}))
        fs_archive["metadata"]["first_stage_vars_retained_per_sol"] = n_fs
        print(
            f"\n--- First-stage archive: {n_fs} variables retained per solution ---",
            flush=True,
        )
        if n_fs == 0:
            print(
                "  WARNING: zero first-stage variables matched.  "
                "Inspect fs_name_set vs. VariableInfo.name values.",
                flush=True,
            )
    else:
        print("  WARNING: no solutions to project", flush=True)

    fs_path = f"{out_prefix}_first_stage.json"
    with open(fs_path, "w") as f:
        json.dump(fs_archive, f, indent=2)
    print(f"  Written {fs_path}", flush=True)

    if full_archive:
        full_path = f"{out_prefix}_full.json"
        print(
            f"\n--- Writing full archive (can be very large) to {full_path} ---",
            flush=True,
        )
        try:
            if hasattr(aos_results, "to_dict"):
                aos_dict = aos_results.to_dict()
            else:
                aos_dict = {"raw": str(aos_results), "n": len(solutions)}
            with open(full_path, "w") as f:
                json.dump(aos_dict, f, indent=0)
            print(f"  Written {full_path}", flush=True)
        except Exception as exc:
            print(f"  Failed to write full archive: {exc}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "General SPAROW + GTEP + OR-TOPAS AOS runner.  "
            "Supports EF and classical Benders (aos-benders generate+filter); "
            "defaults to first-stage-only archives."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sp-module",
        default="sparow_examples.aos_gtep_9bus.aos_single",
        help=(
            "Import path of a module that exports create_sp().  "
            "Typical values: the packages produced by setup_load_scens.py "
            "(benders_single / aos_single) or setup_multiple_load_scens.py "
            "(benders_multi)."
        ),
    )
    p.add_argument(
        "--mode",
        choices=("ef", "benders"),
        default="ef",
        help="Solve path: extensive form or classical Benders.",
    )
    p.add_argument(
        "--aos-method",
        choices=("gurobi_pool", "binary", "hamming"),
        default="gurobi_pool",
        help=(
            "OR-TOPAS AOS kernel.  On the EF path: ordinary AOS.  "
            "On the Benders path: mapped to aos-benders enumeration_method "
            "(hamming → binary).  Candidates are always filtered to true "
            "solutions on the Benders path."
        ),
    )
    p.add_argument(
        "--num-solutions",
        type=int,
        default=3,
        help="Maximum number of alternative solutions to request.",
    )
    p.add_argument(
        "--rel-opt-gap",
        type=float,
        default=0.0,
        help=(
            "Relative optimality gap passed to the AOS / aos-benders kernel.  "
            "Default 0; no parametric sweep is performed."
        ),
    )
    p.add_argument(
        "--max-benders-iterations",
        type=int,
        default=50,
        help="Max Benders iterations when --mode=benders.",
    )
    p.add_argument(
        "--full-archive",
        action="store_true",
        help=(
            "Also write the full-variable JSON archive (can be tens of MB per "
            "solution).  First-stage archive is always written."
        ),
    )
    p.add_argument(
        "--out-prefix",
        default="aos_gtep",
        help="Prefix for output JSON files (<prefix>_first_stage.json, etc.).",
    )
    p.add_argument(
        "--tee",
        action="store_true",
        help="Pass tee=True through to solvers / AOS kernels.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    print("=" * 60, flush=True)
    print(" aos_gtep_runner", flush=True)
    print("=" * 60, flush=True)
    print(f"  sp-module     : {args.sp_module}", flush=True)
    print(f"  mode          : {args.mode}", flush=True)
    print(f"  aos-method    : {args.aos_method}", flush=True)
    print(f"  num-solutions : {args.num_solutions}", flush=True)
    print(f"  rel-opt-gap   : {args.rel_opt_gap}", flush=True)
    print(f"  full-archive  : {args.full_archive}", flush=True)
    print(f"  out-prefix    : {args.out_prefix}", flush=True)

    print("\n--- Loading create_sp and building SP ---", flush=True)
    create_sp = load_create_sp(args.sp_module)
    sp = create_sp()
    print(f"  SP built from {args.sp_module}", flush=True)

    if args.mode == "ef":
        aos_results, model, fs_name_set = run_ef_path(
            sp,
            aos_method=args.aos_method,
            num_solutions=args.num_solutions,
            rel_opt_gap=args.rel_opt_gap,
            tee=args.tee,
        )
    else:
        aos_results, model, fs_name_set = run_benders_path(
            sp,
            aos_method=args.aos_method,
            num_solutions=args.num_solutions,
            rel_opt_gap=args.rel_opt_gap,
            max_iterations=args.max_benders_iterations,
            tee=args.tee,
        )

    metadata = {
        "source_script": "aos_gtep_runner.py",
        "sp_module": args.sp_module,
        "mode": args.mode,
        "aos_method": args.aos_method,
        "num_solutions_requested": args.num_solutions,
        "rel_opt_gap": args.rel_opt_gap,
        "benders_aos": "aos_benders_generate_candidates + aos_benders_filter"
        if args.mode == "benders"
        else "ordinary AOS on EF",
    }
    write_archives(
        aos_results,
        fs_name_set,
        out_prefix=args.out_prefix,
        full_archive=args.full_archive,
        metadata=metadata,
    )

    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
