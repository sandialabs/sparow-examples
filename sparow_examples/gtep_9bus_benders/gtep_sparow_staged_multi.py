"""
GTEP + SPAROW – Multi-scenario staged verification (classical Benders)
=====================================================================

Separate testing path from gtep_sparow_staged.py (single-scenario).

Two scenarios (names match alpha values):
  low_alpha  → alpha=0.90, Probability=0.5
  high_alpha → alpha=1.0,  Probability=0.5

Requires the package produced by setup_multiple_load_scens.py::

    benders_multi/
      low_alpha/    # model copy + data/ + __init__.py
      high_alpha/
      __init__.py   # create_sp() for production sizes

This runner builds the SP with force_minimal size overrides by default so the
first multi-scenario rung stays cheap.  It does not call create_sp() for the
minimal path (that would lock production app_data sizes); it uses the same
model_builder pattern as create_sp but with size_cfg from Stage 1.

driver_gtep.py uses relative imports (from .gtep_model import ...), so scenario
dirs must be importable packages.  Stage 1 prefers package imports and will
write missing __init__.py markers under benders_multi/* as a fallback.

Stage map (same ladder as single)
---------------------------------
0–4  Environment, one concrete model (from low_alpha tree), inventory, trivial SP
5    Two-scenario SPAROW SP + relax_second_stage registration
6    Extensive form (reference)
7    Residual discrete gate with additional_transforms=[relax_second_stage]
8    Benders vs EF (subproblem_transforms=[relax_second_stage])

Usage (from gtep_9bus_benders, after setup_multiple_load_scens.py)
-------------------------------------------------------------------
    python -c "
    from gtep_sparow_staged_multi import run_staged_diagnostics_multi
    run_staged_diagnostics_multi(
        force_minimal=True,
        stop_after_ef=True,
        run_benders=False,
    )
    " 2>&1 | tee gtep_staged_multi_rung1_ef.log

    python -c "
    from gtep_sparow_staged_multi import run_staged_diagnostics_multi
    run_staged_diagnostics_multi(
        force_minimal=True,
        stop_after_ef=False,
        run_benders=True,
        max_benders_iterations=30,
    )
    " 2>&1 | tee gtep_staged_multi_rung1_benders.log
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pyomo.environ as pyo

# Reuse single-scenario stages that do not depend on scenario count
from gtep_sparow_staged import (
    DEFAULT_FIRST_STAGE_NAMES,
    _print,
    stage0_check_environment,
    stage2_inspect_variables,
    stage3_verify_first_stage_names,
    stage4_trivial_sparow_sp,
    stage6_solve_extensive_form,
    stage7_peek_transforms,
    stage8_run_benders,
)

# Multi-scenario definition (names match alpha values)
MULTI_SCENARIOS = [
    {"ID": "low_alpha", "Demand": 1.0, "Probability": 0.5, "alpha": 0.90},
    {"ID": "high_alpha", "Demand": 1.0, "Probability": 0.5, "alpha": 1.0},
]

PACKAGE_NAME = "benders_multi"


# ---------------------------------------------------------------------------
# Stage 1 – one concrete instance (from multi package tree)
# ---------------------------------------------------------------------------

def _ensure_package_roots_on_path():
    """
    driver_gtep uses relative imports (from .gtep_model import ...), so it must
    be loaded as a package member, not via bare file exec.

    Ensure the directory that *contains* benders_multi/ (or model/) is on sys.path.
    """
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()
    roots = [
        cwd,
        here,
        cwd / "sparow_examples",
        here / "sparow_examples",
        cwd.parent,
        here.parent,
    ]
    for root in roots:
        root = root.resolve()
        if root.is_dir() and str(root) not in sys.path:
            sys.path.insert(0, str(root))


def _ensure_scenario_package_markers(pkg_root: Path):
    """Write missing __init__.py so relative imports in driver_gtep work."""
    for init_dir in (
        pkg_root / PACKAGE_NAME,
        pkg_root / PACKAGE_NAME / "low_alpha",
        pkg_root / PACKAGE_NAME / "high_alpha",
    ):
        if init_dir.is_dir():
            init_py = init_dir / "__init__.py"
            if not init_py.is_file():
                init_py.write_text(
                    "# package marker for relative imports in driver_gtep\n"
                )


def _try_import_create_gtep_model_multi():
    """
    Prefer create_gtep_model from benders_multi/<scenario>/ so data/ resolves
    under the multi-scenario package produced by setup_multiple_load_scens.py.

    driver_gtep.py uses relative imports, so we only use package imports
    (not spec_from_file_location without a package context).
    """
    _ensure_package_roots_on_path()

    # Prefer multi package; cover gtep_9bus_benders and aos_gtep_9bus layouts
    candidates = [
        f"{PACKAGE_NAME}.low_alpha.driver_gtep",
        f"{PACKAGE_NAME}.high_alpha.driver_gtep",
        f"sparow_examples.gtep_9bus_benders.{PACKAGE_NAME}.low_alpha.driver_gtep",
        f"sparow_examples.gtep_9bus_benders.{PACKAGE_NAME}.high_alpha.driver_gtep",
        f"gtep_9bus_benders.{PACKAGE_NAME}.low_alpha.driver_gtep",
        f"gtep_9bus_benders.{PACKAGE_NAME}.high_alpha.driver_gtep",
        f"sparow_examples.aos_gtep_9bus.{PACKAGE_NAME}.low_alpha.driver_gtep",
        f"sparow_examples.aos_gtep_9bus.{PACKAGE_NAME}.high_alpha.driver_gtep",
        f"aos_gtep_9bus.{PACKAGE_NAME}.low_alpha.driver_gtep",
        f"aos_gtep_9bus.{PACKAGE_NAME}.high_alpha.driver_gtep",
        # fall backs
        "model.driver_gtep",
        "sparow_examples.gtep_9bus_benders.model.driver_gtep",
        "sparow_examples.aos_gtep_9bus.model.driver_gtep",
        "aos_single.single.driver_gtep",
    ]
    errors: List[str] = []
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, "create_gtep_model", None)
            if fn is None:
                errors.append(f"{mod_name}: no create_gtep_model")
                continue
            driver_file = Path(inspect.getfile(fn)).resolve()
            data_dir = driver_file.parent / "data"
            if data_dir.is_dir():
                _print(f"  create_gtep_model from: {driver_file}")
                _print(f"  data/ resolved to:      {data_dir}")
                return fn
            errors.append(f"{mod_name}: {driver_file} has no sibling data/")
        except Exception as e:
            errors.append(f"{mod_name}: {e}")

    # Package-aware fallback: put parent of benders_multi on path, ensure
    # __init__.py markers, then import as benders_multi.low_alpha.driver_gtep
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()
    for pkg_root in (cwd, here):
        scen_dir = pkg_root / PACKAGE_NAME / "low_alpha"
        driver = scen_dir / "driver_gtep.py"
        data_dir = scen_dir / "data"
        if not driver.is_file() or not data_dir.is_dir():
            continue
        _ensure_scenario_package_markers(pkg_root)
        root_s = str(pkg_root.resolve())
        if root_s not in sys.path:
            sys.path.insert(0, root_s)
        mod_name = f"{PACKAGE_NAME}.low_alpha.driver_gtep"
        try:
            for key in list(sys.modules):
                if key == PACKAGE_NAME or key.startswith(PACKAGE_NAME + "."):
                    del sys.modules[key]
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, "create_gtep_model", None)
            if fn is not None:
                _print(f"  create_gtep_model from: {driver.resolve()}")
                _print(f"  data/ resolved to:      {data_dir.resolve()}")
                return fn
            errors.append(f"{mod_name} (path fallback): no create_gtep_model")
        except Exception as e:
            errors.append(f"{mod_name} (path fallback): {e}")

    raise ImportError(
        "Stage 1 FAILED – could not locate create_gtep_model next to data/.\n"
        f"Run  python setup_multiple_load_scens.py  so {PACKAGE_NAME}/low_alpha/data exists.\n"
        "driver_gtep uses relative imports; the package root that contains "
        f"{PACKAGE_NAME}/ must be on PYTHONPATH (usually the cwd).\n"
        "Also ensure:\n"
        f"  touch {PACKAGE_NAME}/low_alpha/__init__.py {PACKAGE_NAME}/high_alpha/__init__.py\n"
        "Tried:\n  " + "\n  ".join(errors)
    )


def stage1_build_gtep_model_multi(
    *,
    force_minimal: bool = True,
    num_stages: int = 1,
    num_rep_days: int = 1,
    len_rep_days: int = 4,
    num_commit_p: int = 2,
    num_disp: int = 1,
    alpha: float = 1.0,
    flow_model: str = "CP",
    include_commitment: bool = True,
):
    """Build one concrete model for inventory (Stage 1–3). Uses multi package data/."""
    _print("\n=== STAGE 1: Build GTEP model (multi-scenario package tree) ===")

    if force_minimal:
        num_stages = 1
        num_rep_days = 1
        len_rep_days = min(len_rep_days, 4)
        num_commit_p = min(num_commit_p, 2)
        num_disp = min(num_disp, 1)
        flow_model = "CP"
        alpha = 1.0
        _print(
            f"  force_minimal=True → stages={num_stages}, reps={num_rep_days}, "
            f"len_reps={len_rep_days}, commit={num_commit_p}, disp={num_disp}, "
            f"flow={flow_model}"
        )

    create_gtep_model = _try_import_create_gtep_model_multi()
    try:
        model = create_gtep_model(
            num_stages=num_stages,
            num_rep_days=num_rep_days,
            len_rep_days=len_rep_days,
            num_commit_p=num_commit_p,
            num_disp=num_disp,
            alpha=alpha,
            flow_model=flow_model,
            include_commitment=include_commitment,
        )
    except Exception as e:
        raise RuntimeError(
            f"Stage 1 FAILED – create_gtep_model raised:\n  {e}"
        ) from e

    _print(f"  Model type: {type(model)}")
    objs = list(
        model.component_data_objects(pyo.Objective, active=True, descend_into=True)
    )
    if not objs:
        raise RuntimeError("Stage 1 FAILED – no active Objective")
    _print(f"  Active objectives: {[o.name for o in objs]}")

    for sname in (
        "stages",
        "investmentStage",
        "representativePeriods",
        "thermalGenerators",
        "renewableGenerators",
        "storage",
        "transmission",
    ):
        if hasattr(model, sname):
            try:
                n = len(getattr(model, sname))
            except TypeError:
                n = "?"
            _print(f"  Component {sname}: length {n}")

    n_bin = sum(
        1
        for v in model.component_data_objects(pyo.Var, active=True, descend_into=True)
        if v.domain is pyo.Binary or "binary" in str(v.domain).lower()
    )
    _print(f"  Total active Binary VarData (after bigM): {n_bin}")
    _print("  Stage 1 PASSED")
    return model, {
        "num_stages": num_stages,
        "num_rep_days": num_rep_days,
        "len_rep_days": len_rep_days,
        "num_commit_p": num_commit_p,
        "num_disp": num_disp,
        "alpha": alpha,
        "flow_model": flow_model,
        "include_commitment": include_commitment,
    }


# ---------------------------------------------------------------------------
# Stage 5 – two-scenario SPAROW SP
# ---------------------------------------------------------------------------

def stage5_gtep_sparow_sp_multi(
    size_cfg: dict,
    *,
    first_stage_names: Sequence[str],
    scenarios: Optional[List[dict]] = None,
    register_relax_second_stage: bool = True,
    package_name: str = PACKAGE_NAME,
):
    """
    Build a two-scenario SPAROW SP.

    model_builder imports create_gtep_model from
    {package_name}.{ID} (and gtep_9bus_benders / aos_gtep_9bus variants)
    so each scenario keeps its own data/ tree.

    Size knobs come from size_cfg (force_minimal), not from create_sp() app_data.
    """
    _print("\n=== STAGE 5: SPAROW SP with GTEP model_builder (MULTI-SCENARIO) ===")
    from sparow.sp import stochastic_program
    from sparow.sp.util import relax_second_stage

    _ensure_package_roots_on_path()
    for pkg_root in (Path.cwd(), Path(__file__).resolve().parent):
        if (pkg_root / package_name).is_dir():
            _ensure_scenario_package_markers(pkg_root)

    if scenarios is None:
        # Only scenario-specific keys here. Size knobs live in app_data only.
        # SPAROW merges app_data into builder data and asserts no key collisions.
        scenarios = [
            {
                "ID": s["ID"],
                "Demand": s.get("Demand", 1.0),
                "Probability": s["Probability"],
                "alpha": s["alpha"],
            }
            for s in MULTI_SCENARIOS
        ]
    else:
        # Strip any size keys a caller may have put on scenario dicts
        clean = []
        size_keys = {
            "stages",
            "num_reps",
            "len_reps",
            "num_commit",
            "num_dispatch",
            "flow_model",
            "include_commitment",
        }
        for s in scenarios:
            clean.append({k: v for k, v in s.items() if k not in size_keys})
        scenarios = clean

    app_data = {
        "stages": size_cfg["num_stages"],
        "num_reps": size_cfg["num_rep_days"],
        "len_reps": size_cfg["len_rep_days"],
        "num_commit": size_cfg["num_commit_p"],
        "num_dispatch": size_cfg["num_disp"],
    }

    _print(
        f"  app_data sizes: stages={app_data['stages']}, "
        f"reps={app_data['num_reps']}, len_reps={app_data['len_reps']}, "
        f"commit={app_data['num_commit']}, disp={app_data['num_dispatch']}"
    )
    _print(f"  Scenarios ({len(scenarios)}):")
    for s in scenarios:
        _print(
            f"    ID={s['ID']!r}  alpha={s.get('alpha')}  P={s.get('Probability')}"
        )

    model_data = {"data": {}, "scenarios": scenarios}

    def gtep_builder(data, args):
        num_stages = int(data.get("stages", size_cfg["num_stages"]))
        num_rep_days = int(data.get("num_reps", size_cfg["num_rep_days"]))
        len_rep_days = int(data.get("len_reps", size_cfg["len_rep_days"]))
        num_commit_p = int(data.get("num_commit", size_cfg["num_commit_p"]))
        num_disp = int(data.get("num_dispatch", size_cfg["num_disp"]))
        alpha = float(data.get("alpha", 1.0))
        flow_model = data.get("flow_model", size_cfg.get("flow_model", "CP"))
        include_commitment = data.get(
            "include_commitment", size_cfg.get("include_commitment", True)
        )
        scen_id = data.get("ID", "low_alpha")

        mod = None
        for mod_name in (
            f"{package_name}.{scen_id}",
            f"sparow_examples.gtep_9bus_benders.{package_name}.{scen_id}",
            f"gtep_9bus_benders.{package_name}.{scen_id}",
            f"sparow_examples.aos_gtep_9bus.{package_name}.{scen_id}",
            f"aos_gtep_9bus.{package_name}.{scen_id}",
        ):
            try:
                mod = importlib.import_module(mod_name)
                break
            except ImportError:
                continue
        if mod is None or not hasattr(mod, "create_gtep_model"):
            create_fn = _try_import_create_gtep_model_multi()
            return create_fn(
                num_stages=num_stages,
                num_rep_days=num_rep_days,
                len_rep_days=len_rep_days,
                num_commit_p=num_commit_p,
                num_disp=num_disp,
                alpha=alpha,
                flow_model=flow_model,
                include_commitment=include_commitment,
            )
        return mod.create_gtep_model(
            num_stages=num_stages,
            num_rep_days=num_rep_days,
            len_rep_days=len_rep_days,
            num_commit_p=num_commit_p,
            num_disp=num_disp,
            alpha=alpha,
            flow_model=flow_model,
            include_commitment=include_commitment,
        )

    sp = stochastic_program(first_stage_variables=list(first_stage_names))
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(model_data=model_data, model_builder=gtep_builder)

    if register_relax_second_stage:
        sp.add_transformation(relax_second_stage)
        _print(
            "  Registered relax_second_stage "
            "(residual second-stage discrete → continuous)"
        )

    _print(f"  default_model: {sp.default_model!r}")
    _print(f"  bundles: {list(sp.bundles) if sp.bundles else None}")
    if not sp.bundles:
        raise RuntimeError("Stage 5 FAILED – no bundles (expected 2 scenarios)")
    if len(sp.bundles) < 2:
        _print(
            f"  WARNING: expected 2 bundles, got {len(sp.bundles)}. "
            "Check model_data scenarios."
        )

    b0 = next(iter(sp.bundles))
    _print(f"  Building EF for bundle {b0} (map population) ...")
    try:
        sp.create_bundle_EF(b=b0, w=None, x_bar=None, rho=None, cached=False)
    except Exception as e:
        raise RuntimeError(
            f"Stage 5 FAILED – create_bundle_EF raised:\n  {e}"
        ) from e

    if not hasattr(sp, "int_to_FirstStageVar") or b0 not in sp.int_to_FirstStageVar:
        raise RuntimeError(
            "Stage 5 FAILED – int_to_FirstStageVar missing after create_bundle_EF"
        )

    fs_map = sp.int_to_FirstStageVar[b0]
    _print(f"  int_to_FirstStageVar[{b0}]: {len(fs_map)} variables")
    if len(fs_map) == 0:
        raise RuntimeError("Stage 5 FAILED – empty first-stage map")

    for i, v in list(fs_map.items())[:6]:
        _print(f"    [{i}] {v.name}  domain={v.domain}")

    non_bin = [
        v
        for v in fs_map.values()
        if v.domain is not pyo.Binary
        and "binary" not in str(v.domain).lower()
        and "integer" not in str(v.domain).lower()
    ]
    if non_bin:
        _print(
            f"  NOTE: {len(non_bin)} first-stage vars are continuous "
            "(expected for renewable MW)"
        )

    _print("  Stage 5 PASSED (multi-scenario)")
    return sp


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_staged_diagnostics_multi(
    *,
    force_minimal: bool = True,
    first_stage_names: Optional[Sequence[str]] = None,
    scenarios: Optional[List[dict]] = None,
    stop_after_ef: bool = True,
    run_benders: bool = False,
    ef_solver: str = "gurobi",
    ef_time_limit: float = 600,
    max_benders_iterations: int = 50,
    num_stages: int = 1,
    num_rep_days: int = 1,
    len_rep_days: int = 4,
    num_commit_p: int = 2,
    num_disp: int = 1,
    flow_model: str = "CP",
    include_commitment: bool = True,
    skip_trivial_sp: bool = False,
    package_name: str = PACKAGE_NAME,
):
    """
    Multi-scenario staged path.  Independent of run_staged_diagnostics (single).

    Default scenarios::

        low_alpha  alpha=0.90  P=0.5
        high_alpha alpha=1.0   P=0.5
    """
    _print("=" * 72)
    _print("GTEP SPAROW staged diagnostics – MULTI-SCENARIO path")
    _print(f"  package={package_name}")
    _print("=" * 72)

    stage0_check_environment()

    model, size_cfg = stage1_build_gtep_model_multi(
        force_minimal=force_minimal,
        num_stages=num_stages,
        num_rep_days=num_rep_days,
        len_rep_days=len_rep_days,
        num_commit_p=num_commit_p,
        num_disp=num_disp,
        flow_model=flow_model,
        include_commitment=include_commitment,
    )

    stage2_inspect_variables(model)

    if first_stage_names is None:
        first_stage_names = list(DEFAULT_FIRST_STAGE_NAMES)
        _print(
            f"\n  Using DEFAULT_FIRST_STAGE_NAMES ({len(first_stage_names)} patterns)"
        )

    stage3_verify_first_stage_names(model, first_stage_names)

    if not skip_trivial_sp:
        stage4_trivial_sparow_sp()
    else:
        _print("\n=== STAGE 4: skipped (skip_trivial_sp=True) ===")

    if scenarios is None:
        scenarios = [dict(s) for s in MULTI_SCENARIOS]

    sp = stage5_gtep_sparow_sp_multi(
        size_cfg,
        first_stage_names=first_stage_names,
        scenarios=scenarios,
        register_relax_second_stage=True,
        package_name=package_name,
    )

    ef_ref = stage6_solve_extensive_form(
        sp,
        solver_name=ef_solver,
        time_limit=ef_time_limit,
    )

    if stop_after_ef and not run_benders:
        _print("\n*** MULTI-SCENARIO staged diagnostics complete through EF. ***")
        _print(f"*** EF reference objective = {ef_ref['objective']:.6g} ***")
        _print(
            "*** Note: EF may rebundle scenarios into a single 'bundle'. ***"
        )
        _print("*** Re-run with run_benders=True only after this stage is clean. ***")
        return {"sp": sp, "ef_reference": ef_ref, "size_cfg": size_cfg, "path": "multi"}

    # -----------------------------------------------------------------
    # Rebuild SP for Stages 7–8.
    #
    # ExtensiveFormSolver uses a single_bundle scheme that collapses
    # low_alpha + high_alpha into one bundle with two scenarios.
    # Classical Benders (_transform_to_master_model /
    # _transform_to_subproblem_model) asserts len(bundle.scenarios) == 1.
    # A fresh Stage-5 SP restores one-scenario-per-bundle structure while
    # keeping the EF objective as the reference.
    # -----------------------------------------------------------------
    _print("\n=== Rebuild SP for Benders (one scenario per bundle) ===")
    _print(
        "  EF may have rebundled scenarios; classical Benders needs "
        "one scenario per bundle."
    )
    sp_benders = stage5_gtep_sparow_sp_multi(
        size_cfg,
        first_stage_names=first_stage_names,
        scenarios=scenarios,
        register_relax_second_stage=True,
        package_name=package_name,
    )
    _print(f"  Benders SP bundles: {list(sp_benders.bundles)}")
    for bname in sp_benders.bundles:
        n_scen = len(sp_benders.bundles[bname].scenarios)
        _print(f"    bundle {bname!r}: {n_scen} scenario(s)")
        if n_scen != 1:
            _print(
                f"    WARNING: bundle {bname!r} has {n_scen} scenarios; "
                "Benders transform will assert."
            )

    residual_info = stage7_peek_transforms(sp_benders)

    if not run_benders:
        _print(
            "\n*** MULTI EF + transform peek done. "
            "Set run_benders=True for Stage 8. ***"
        )
        return {
            "sp": sp_benders,
            "sp_ef": sp,
            "ef_reference": ef_ref,
            "size_cfg": size_cfg,
            "residual": residual_info,
            "path": "multi",
        }

    rb = residual_info.get("residual_binary")
    if rb is not None and rb > 0:
        _print(
            "\n*** WARNING: residual Binary > 0 after "
            "additional_transforms=[relax_second_stage]. ***\n"
            "*** Dual cuts may be invalid if the subproblem is not LP. ***"
        )

    benders_out = stage8_run_benders(
        sp_benders,
        ef_reference=ef_ref,
        max_iterations=max_benders_iterations,
    )
    return {
        "sp": sp_benders,
        "sp_ef": sp,
        "ef_reference": ef_ref,
        "size_cfg": size_cfg,
        "residual": residual_info,
        "benders": benders_out,
        "path": "multi",
    }


if __name__ == "__main__":
    run_staged_diagnostics_multi(
        force_minimal=True,
        stop_after_ef=True,
        run_benders=False,
        ef_solver="gurobi",
        ef_time_limit=600,
    )
