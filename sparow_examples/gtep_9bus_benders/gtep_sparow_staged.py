"""
GTEP + SPAROW – Staged Verification Path for Classical Benders
==============================================================

Goal
----
Prove that a GTEP (Generation & Transmission Expansion Planning) model can be:
  1. Wrapped as a SPAROW StochasticProgram
  2. Solved as an extensive form (EF) for a trusted reference objective
  3. Decomposed with classical Benders (SPAROW → OR-Topas standard_lp + cuts)
  4. Shown to recover the same objective as the EF (within tolerance)

Do this with staged diagnostics so each failure has a clear signature
before blaming Benders cuts.

Residual discrete variables are handled by the official SPAROW utility
    from sparow.sp.util import relax_second_stage
exactly as already used in aos_single_ef.py.

- EF path (Stage 5–6): sp.add_transformation(relax_second_stage)
- Benders path (Stage 7–8): BendersSolver additional_transforms /
  subproblem_transforms=[relax_second_stage] (new SPAROW hook). Master does
  NOT receive residual relaxation so first-stage stay discrete.

After the transform, every non-first-stage Binary/Integer is continuous so
classical dual cuts remain valid.

Stage map
---------
0. Environment & imports
1. Build a concrete (preferably minimal) GTEP model via create_gtep_model
2. Inventory binary / integer variables; emit heuristic first-stage candidates
3. Verify the concrete first-stage name list (investment indicators)
4. Minimal SPAROW SP with a trivial builder (plumbing check)
5. Domain model_builder → SPAROW SP; register relax_second_stage; inspect maps
6. **Solve the extensive form** – reference objective (critical control point)
7. Peek at master / subproblem transforms; residual Binary count must be ~0
8. Benders solve + automatic comparison against the EF objective

Do not attempt Stage 8 until Stage 6 has produced a finite EF objective and
Stage 7 reports residual discrete ≈ 0.

Minimal first-rung recipe (recommended after any change)
-------------------------------------------------------
    run_staged_diagnostics(
        force_minimal=True,          # stages=1, 1 short rep-day, short commit/disp, CP
        stop_after_ef=True,
        run_benders=False,
    )
Only set run_benders=True after the EF objective is clean and residual Binary=0.
"""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyomo.environ as pyo

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def _print(*args, **kwargs):
    """print with forced flush so stage banners and diagnostics appear immediately.

    Plain print() can stay buffered under redirection, pipes, or heavy solver
    output. The EGRET ladder required this discipline so the console is never
    silent while a long EF / Benders / transform step is running: if the
    terminal is quiet the process is inside a solver or transform call, not
    stuck at a stage boundary.
    """
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Canonical first-stage list (investment indicators)
# Matches the patterns used in setup_load_scens.py / aos_single create_sp.
# SPAROW resolves these via ComponentUID (supports [*]).
# ---------------------------------------------------------------------------
DEFAULT_FIRST_STAGE_NAMES: List[str] = [
    "investmentStage[*].renewableOperational[*]",
    "investmentStage[*].renewableInstalled[*]",
    "investmentStage[*].renewableRetired[*]",
    "investmentStage[*].renewableExtended[*]",
    "investmentStage[*].renewableDisabled[*]",
    "investmentStage[*].genOperational[*].binary_indicator_var",
    "investmentStage[*].genInstalled[*].binary_indicator_var",
    "investmentStage[*].genRetired[*].binary_indicator_var",
    "investmentStage[*].genDisabled[*].binary_indicator_var",
    "investmentStage[*].genExtended[*].binary_indicator_var",
    "investmentStage[*].branchOperational[*].binary_indicator_var",
    "investmentStage[*].branchInstalled[*].binary_indicator_var",
    "investmentStage[*].branchRetired[*].binary_indicator_var",
    "investmentStage[*].branchDisabled[*].binary_indicator_var",
    "investmentStage[*].branchExtended[*].binary_indicator_var",
    "investmentStage[*].storOperational[*].binary_indicator_var",
    "investmentStage[*].storInstalled[*].binary_indicator_var",
    "investmentStage[*].storRetired[*].binary_indicator_var",
    "investmentStage[*].storDisabled[*].binary_indicator_var",
    "investmentStage[*].storExtended[*].binary_indicator_var",
]


# ---------------------------------------------------------------------------
# Stage 0 – Environment
# ---------------------------------------------------------------------------

def stage0_check_environment():
    """
    Verify that the required packages can be imported.
    Clear failure: ImportError with the missing package name.
    """
    _print("\n=== STAGE 0: Environment ===")
    missing = []

    try:
        import pyomo
        from pyomo.environ import ConcreteModel
        _print(f"  Pyomo OK  ({pyomo.__file__})")
    except ImportError as e:
        missing.append(f"pyomo ({e})")

    try:
        import sparow
        from sparow.sp import stochastic_program
        from sparow.sp.util import relax_second_stage
        _print(f"  SPAROW OK ({sparow.__file__})")
        _print(f"  relax_second_stage importable")
    except ImportError as e:
        missing.append(f"sparow / relax_second_stage ({e})")

    try:
        import or_topas
        _print(f"  OR-Topas OK")
    except ImportError as e:
        _print(f"  OR-Topas not found (needed for Benders / AOS): {e}")
        # not fatal for Stages 0–6 if only EF is required

    try:
        import egret
        from egret.data.model_data import ModelData
        _print(f"  EGRET OK  ({egret.__file__})")
    except ImportError as e:
        missing.append(f"egret ({e})")

    if missing:
        raise ImportError(
            "Stage 0 FAILED – missing packages:\n  " + "\n  ".join(missing)
        )
    _print("  Stage 0 PASSED")
    return True


# ---------------------------------------------------------------------------
# Stage 1 – Build one concrete GTEP instance
# ---------------------------------------------------------------------------

def _try_import_create_gtep_model():
    """
    Locate create_gtep_model from a package/file whose directory has a sibling
    data/ folder.

    Layout reality (aos_gtep_9bus code-gen paradigm)
    -----------------------------------------------
    data/ does NOT live next to aos_single_ef.py or this staged script.
    It lives under the model tree that gets copied by setup_load_scens:

        model/data/                         # template
        aos_single/single/data/             # after setup_load_scens.py

    driver_gtep resolves data via Path(__file__).parent / "data", so the
    import MUST land on a driver_gtep.py that still has its sibling data/.
    Prefer a generated scenario package; fall back to the template model/.
    """
    import importlib
    import inspect

    candidates = [
        # Generated single-scenario package (preferred after setup_load_scens)
        "sparow_examples.aos_gtep_9bus.aos_single.single.driver_gtep",
        "aos_single.single.driver_gtep",
        # Template model tree
        "sparow_examples.aos_gtep_9bus.model.driver_gtep",
        "model.driver_gtep",
        # Relative / same-package fallbacks
        ".aos_single.single.driver_gtep",
        ".model.driver_gtep",
        "driver_gtep",
    ]
    errors = []
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, "create_gtep_model", None)
            if fn is None:
                errors.append(f"{mod_name}: no create_gtep_model attribute")
                continue
            driver_file = Path(inspect.getfile(fn)).resolve()
            data_dir = driver_file.parent / "data"
            if data_dir.is_dir():
                _print(f"  create_gtep_model from: {driver_file}")
                _print(f"  data/ resolved to:      {data_dir}")
                return fn
            errors.append(f"{mod_name}: found {driver_file} but no sibling data/")
        except Exception as e:
            errors.append(f"{mod_name}: {e}")

    # File-based fallback relative to this script / cwd
    here = Path(__file__).resolve().parent
    for rel in (
        here / "aos_single" / "single" / "driver_gtep.py",
        here / "model" / "driver_gtep.py",
        Path.cwd() / "aos_single" / "single" / "driver_gtep.py",
        Path.cwd() / "model" / "driver_gtep.py",
        here.parent / "model" / "driver_gtep.py",
    ):
        if not rel.is_file():
            continue
        data_dir = rel.parent / "data"
        if not data_dir.is_dir():
            errors.append(f"file {rel}: no sibling data/")
            continue
        import importlib.util
        spec = importlib.util.spec_from_file_location("local_driver_gtep", rel)
        mod = importlib.util.module_from_spec(spec)
        try:
            sys.path.insert(0, str(rel.parent))
            spec.loader.exec_module(mod)
            if hasattr(mod, "create_gtep_model"):
                _print(f"  create_gtep_model loaded from: {rel}")
                _print(f"  data/ resolved to:            {data_dir}")
                return mod.create_gtep_model
        except Exception as e:
            errors.append(f"file {rel}: {e}")

    raise ImportError(
        "Stage 1 FAILED – could not locate create_gtep_model next to a data/ directory.\n"
        "data/ lives under the model tree (model/data/ or, after setup_load_scens,\n"
        "aos_single/single/data/), NOT next to this staged script.\n\n"
        "Tried:\n  " + "\n  ".join(errors) + "\n\n"
        "Fix: run  python setup_load_scens.py  once so aos_single/single/ exists\n"
        "with its data/ copy, then ensure that package is importable, OR place\n"
        "this script so that model/driver_gtep.py + model/data/ resolve."
    )


def stage1_build_gtep_model(
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
    storage: bool = True,
    transmission: bool = True,
):
    """
    Build one concrete GTEP model with the same API the production path uses.

    force_minimal=True overrides the size knobs to a fast validation configuration.

    Success signature
    -----------------
    - Returns a ConcreteModel with at least one active Objective.
    - GDP bigM has already been applied (binaries exist).
    """
    _print("\n=== STAGE 1: Build GTEP model ===")

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

    create_gtep_model = _try_import_create_gtep_model()

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
            f"Stage 1 FAILED – create_gtep_model raised:\n  {e}\n"
            "Check that the relative data/ directory (Prescient/RTS-style) "
            "is present next to driver_gtep.py."
        ) from e

    _print(f"  Model type: {type(model)}")
    _print(f"  Model name: {getattr(model, 'name', None)}")

    objs = list(model.component_data_objects(pyo.Objective, active=True, descend_into=True))
    if not objs:
        raise RuntimeError(
            "Stage 1 FAILED – GTEP model has no active Objective. "
            "SPAROW cost splitting and Benders will not work."
        )
    _print(f"  Active objectives: {[o.name for o in objs]}")

    # Key hierarchical sets / blocks if present
    for sname in ("stages", "investmentStage", "representativePeriods", "thermalGenerators",
                  "renewableGenerators", "storage", "transmission", "Buses"):
        if hasattr(model, sname):
            s = getattr(model, sname)
            try:
                n = len(s)
            except TypeError:
                n = "?"
            _print(f"  Component {sname}: length {n}")
        else:
            # may live under investmentStage blocks
            pass

    n_bin = sum(
        1 for v in model.component_data_objects(pyo.Var, active=True, descend_into=True)
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
# Stage 2 – Inventory discrete structure
# ---------------------------------------------------------------------------

def stage2_inspect_variables(model: pyo.ConcreteModel):
    """
    Print every binary/integer variable component and a sample of indices.

    Success signature
    -----------------
    - At least one Binary component (investment indicators after bigM).
    - Printed table of (component name, domain, #indices, sample indices).
    """
    _print("\n=== STAGE 2: Inspect binary / integer variables ===")

    binary_comps = []
    integer_comps = []
    continuous_sample = []

    for comp in model.component_objects(pyo.Var, descend_into=True):
        try:
            sample = next(comp.values()) if comp.is_indexed() else comp
        except StopIteration:
            continue
        domain = sample.domain
        n = len(comp) if comp.is_indexed() else 1
        sample_idx = list(comp.index_set())[:3] if comp.is_indexed() else None

        info = {
            "name": comp.name,
            "domain": str(domain),
            "n": n,
            "sample_indices": sample_idx,
            "is_indexed": comp.is_indexed(),
        }
        if domain is pyo.Binary or str(domain).lower() == "binary":
            binary_comps.append(info)
        elif domain is pyo.Integers or "integer" in str(domain).lower():
            integer_comps.append(info)
        else:
            if len(continuous_sample) < 8:
                continuous_sample.append(info)

    _print("  Binary components:")
    if not binary_comps:
        _print("    *** NONE FOUND ***")
        raise RuntimeError(
            "Stage 2 FAILED – no Binary variables after gdp.bigm. "
            "Did the GDP transform fail, or was the model built relaxed?"
        )
    for b in binary_comps:
        _print(f"    {b['name']:70s}  n={b['n']:5d}  sample_idx={b['sample_indices']}")

    _print("  Integer (non-binary) components:")
    if not integer_comps:
        _print("    (none)")
    for b in integer_comps:
        _print(f"    {b['name']:70s}  n={b['n']:5d}  sample_idx={b['sample_indices']}")

    _print("  Continuous sample (first few):")
    for b in continuous_sample:
        _print(f"    {b['name']:70s}  n={b['n']:5d}")

    # Heuristic: investment-stage binaries + continuous renewable MW vars
    keywords = (
        "investmentstage", "genoperational", "geninstalled", "genretired",
        "gendisabled", "genextended", "branch", "stor", "renewable",
        "binary_indicator",
    )
    suggested = [
        b["name"] for b in binary_comps
        if any(k in b["name"].lower() for k in keywords)
    ]
    _print(f"\n  Suggested first_stage candidates (heuristic, binary only): {len(suggested)} names")
    for s in suggested[:12]:
        _print(f"    {s}")
    if len(suggested) > 12:
        _print(f"    ... +{len(suggested)-12} more")

    _print("  Stage 2 PASSED – inspect the list above before continuing")
    return binary_comps, integer_comps, suggested


# ---------------------------------------------------------------------------
# Stage 3 – Verify a concrete first-stage name list
# ---------------------------------------------------------------------------

def stage3_verify_first_stage_names(
    model: pyo.ConcreteModel,
    first_stage_names: Sequence[str],
):
    """
    Confirm that the first-stage patterns resolve against the model.

    Because SPAROW uses ComponentUID (which accepts [*]), we do not require
    every pattern to resolve under a plain getattr; we only require that at
    least some Binary variables exist whose names are consistent with the
    investment-stage hierarchy.  A deeper check happens in Stage 5 after
    SPAROW builds int_to_FirstStageVar.

    Success signature
    -----------------
    - first_stage_names is non-empty
    - model still has Binary variables
    """
    _print("\n=== STAGE 3: Verify first-stage name list ===")
    _print(f"  Candidate patterns ({len(first_stage_names)}):")
    for n in first_stage_names:
        _print(f"    {n}")

    if not first_stage_names:
        raise RuntimeError("Stage 3 FAILED – first_stage_names is empty")

    # Light structural check: at least one investmentStage block exists
    has_invest = any(
        "investmentstage" in (c.name or "").lower()
        for c in model.component_objects(pyo.Block, descend_into=True)
    ) or hasattr(model, "investmentStage")
    if not has_invest:
        _print("  WARNING: no component name containing 'investmentStage' found. "
              "Patterns may not resolve.")

    n_bin = sum(
        1 for v in model.component_data_objects(pyo.Var, active=True, descend_into=True)
        if v.domain is pyo.Binary or "binary" in str(v.domain).lower()
    )
    _print(f"  Model still has {n_bin} active Binary VarData objects")
    if n_bin == 0:
        raise RuntimeError("Stage 3 FAILED – no Binary variables left on the model")

    _print("  Stage 3 PASSED (full map population is verified in Stage 5)")
    return list(first_stage_names)


# ---------------------------------------------------------------------------
# Stage 4 – Minimal SPAROW SP with a trivial builder
# ---------------------------------------------------------------------------

def stage4_trivial_sparow_sp():
    """
    Build a SPAROW StochasticProgram using a trivial newsvendor-style builder.
    Confirms SPAROW plumbing before involving the GTEP model_builder.
    """
    _print("\n=== STAGE 4: Trivial SPAROW SP (newsvendor-style) ===")
    from sparow.sp import stochastic_program

    def builder(data, args=None):
        m = pyo.ConcreteModel(str(data.get("ID", "s")))
        m.x = pyo.Var(within=pyo.NonNegativeReals)
        d = data.get("d", 50)
        c, b, h = 1.0, 1.5, 0.1
        m.y = pyo.Var()
        m.greater = pyo.Constraint(expr=m.y >= (c - b) * m.x + b * d)
        m.less = pyo.Constraint(expr=m.y >= (c + h) * m.x - h * d)
        m.o = pyo.Objective(expr=m.y)
        return m

    model_data = {
        "scenarios": [
            {"ID": 1, "d": 15},
            {"ID": 2, "d": 60},
            {"ID": 3, "d": 72},
        ]
    }

    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data={})
    # Do NOT pass name=... – leave default (None). Same pattern the Benders path expects.
    sp.initialize_model(
        model_data=model_data,
        model_builder=builder,
        default=True,
    )

    _print(f"  default_model: {sp.default_model!r}")
    _print(f"  bundles: {list(sp.bundles) if sp.bundles is not None else None}")
    if sp.bundles is None or len(sp.bundles) == 0:
        raise RuntimeError(
            "Stage 4 FAILED – no bundles after initialize_model. "
            "Check SPAROW bundling defaults / version."
        )

    b0 = next(iter(sp.bundles))
    try:
        m = sp.create_bundle_EF(b=b0, w=None, x_bar=None, rho=None, cached=False)
        _print(f"  create_bundle_EF succeeded for bundle {b0}")
    except Exception as e:
        raise RuntimeError(
            f"Stage 4 FAILED – create_bundle_EF raised: {e}"
        ) from e

    if hasattr(sp, "int_to_FirstStageVar") and b0 in sp.int_to_FirstStageVar:
        fs_map = sp.int_to_FirstStageVar[b0]
        _print(f"  int_to_FirstStageVar[{b0}]: {len(fs_map)} entries")
    else:
        _print("  WARNING: int_to_FirstStageVar not yet populated (may be lazy)")

    _print("  Stage 4 PASSED")
    return sp


# ---------------------------------------------------------------------------
# Stage 5 – GTEP model_builder → SPAROW SP + relax_second_stage
# ---------------------------------------------------------------------------

def stage5_gtep_sparow_sp(
    size_cfg: dict,
    first_stage_names: Sequence[str],
    scenario_scales: Optional[Dict[Any, float]] = None,
    *,
    register_relax_second_stage: bool = True,
):
    """
    Construct a SPAROW SP whose model_builder calls create_gtep_model.

    Always registers relax_second_stage (unless explicitly disabled) so that
    residual second-stage discretes become continuous before EF / Benders.

    Success signature
    -----------------
    - bundles non-empty
    - int_to_FirstStageVar non-empty for each bundle
    - create_bundle_EF succeeds
    """
    _print("\n=== STAGE 5: SPAROW SP with GTEP model_builder ===")
    from sparow.sp import stochastic_program
    from sparow.sp.util import relax_second_stage

    create_gtep_model = _try_import_create_gtep_model()

    if scenario_scales is None:
        scenario_scales = {"single": 1.0}

    scenarios = []
    for sid, scale in scenario_scales.items():
        scenarios.append({
            "ID": sid,
            "Demand": 1.0,
            "Probability": 1.0 / max(len(scenario_scales), 1),
            "alpha": float(scale) * size_cfg.get("alpha", 1.0),
            **{k: size_cfg[k] for k in (
                "num_stages", "num_rep_days", "len_rep_days",
                "num_commit_p", "num_disp", "flow_model", "include_commitment",
            ) if k in size_cfg},
        })

    # SPAROW expects stages / num_reps etc. either in app_data or scenario data.
    # We put size knobs in both places for robustness.
    app_data = {
        "stages": size_cfg.get("num_stages", 1),
        "num_reps": size_cfg.get("num_rep_days", 1),
        "len_reps": size_cfg.get("len_rep_days", 4),
        "num_commit": size_cfg.get("num_commit_p", 2),
        "num_dispatch": size_cfg.get("num_disp", 1),
    }
    model_data = {"data": {}, "scenarios": scenarios}

    def gtep_builder(data, args=None):
        # Prefer values carried on the scenario; fall back to size_cfg / app defaults
        num_stages = int(data.get("num_stages", app_data["stages"]))
        num_rep_days = int(data.get("num_rep_days", app_data["num_reps"]))
        len_rep_days = int(data.get("len_rep_days", app_data["len_reps"]))
        num_commit_p = int(data.get("num_commit_p", app_data["num_commit"]))
        num_disp = int(data.get("num_disp", app_data["num_dispatch"]))
        alpha = float(data.get("alpha", 1.0))
        flow_model = data.get("flow_model", size_cfg.get("flow_model", "CP"))
        include_commitment = data.get(
            "include_commitment", size_cfg.get("include_commitment", True)
        )

        m = create_gtep_model(
            num_stages=num_stages,
            num_rep_days=num_rep_days,
            len_rep_days=len_rep_days,
            num_commit_p=num_commit_p,
            num_disp=num_disp,
            alpha=alpha,
            flow_model=flow_model,
            include_commitment=include_commitment,
        )
        # Ensure a canonical objective name that SPAROW transforms tolerate
        objs = list(m.component_data_objects(pyo.Objective, active=True))
        if objs and objs[0].name not in ("o", "obj"):
            m.o = pyo.Objective(expr=objs[0].expr, sense=objs[0].sense)
            objs[0].deactivate()
        return m

    sp = stochastic_program(first_stage_variables=list(first_stage_names))
    sp.initialize_application(app_data=app_data)
    # Do NOT pass name=... – leave default (None)
    sp.initialize_model(
        model_data=model_data,
        model_builder=gtep_builder,
        default=True,
    )

    if register_relax_second_stage:
        sp.add_transformation(relax_second_stage)
        _print("  Registered relax_second_stage (residual second-stage discrete → continuous)")

    _print(f"  default_model: {sp.default_model!r}")
    _print(f"  bundles: {list(sp.bundles) if sp.bundles else None}")
    if not sp.bundles:
        raise RuntimeError("Stage 5 FAILED – no bundles")

    b0 = next(iter(sp.bundles))
    _print(f"  Building EF for bundle {b0} ...")
    try:
        m = sp.create_bundle_EF(b=b0, w=None, x_bar=None, rho=None, cached=False)
    except Exception as e:
        raise RuntimeError(
            f"Stage 5 FAILED – create_bundle_EF / GTEP builder raised:\n  {e}"
        ) from e

    if not hasattr(sp, "int_to_FirstStageVar") or b0 not in sp.int_to_FirstStageVar:
        raise RuntimeError(
            "Stage 5 FAILED – int_to_FirstStageVar missing after create_bundle_EF. "
            "SPAROW did not recognise any of the first_stage_variables patterns "
            f"{list(first_stage_names)}.  Re-run Stage 2 and correct the name list."
        )

    fs_map = sp.int_to_FirstStageVar[b0]
    _print(f"  int_to_FirstStageVar[{b0}]: {len(fs_map)} variables")
    if len(fs_map) == 0:
        raise RuntimeError(
            "Stage 5 FAILED – first-stage map is empty. "
            "The patterns in first_stage_variables were not found as Var components "
            "in the GTEP model.  Check Stage 2 output and the [*] ComponentUID patterns."
        )

    for i, v in list(fs_map.items())[:8]:
        _print(f"    [{i}] {v.name}  domain={v.domain}")

    non_bin = [
        v for v in fs_map.values()
        if v.domain is not pyo.Binary and "binary" not in str(v.domain).lower()
        and "integer" not in str(v.domain).lower()
    ]
    # Continuous renewable MW vars are expected in the first-stage list; warn only
    # if something unexpected appears.
    if non_bin:
        _print(f"  NOTE: {len(non_bin)} first-stage vars are continuous (expected for renewable MW)")

    _print("  Stage 5 PASSED")
    return sp


# ---------------------------------------------------------------------------
# Stage 6 – Extensive Form solve (reference)
# ---------------------------------------------------------------------------

def stage6_solve_extensive_form(
    sp,
    solver_name: str = "gurobi",
    solver_options: Optional[dict] = None,
    time_limit: Optional[float] = 600,
):
    """
    Solve the extensive form.  Critical control point before any Benders work.

    Success signature
    -----------------
    - Finite numeric objective recovered.
    """
    _print("\n=== STAGE 6: Extensive Form solve (reference) ===")
    from sparow.ef import ExtensiveFormSolver

    if solver_options is None:
        solver_options = {}
    if time_limit is not None:
        solver_options.setdefault("TimeLimit", time_limit)
        solver_options.setdefault("timelimit", time_limit)

    ef_solver = ExtensiveFormSolver()
    ef_solver.set_options(solver=solver_name, solver_options=solver_options)

    _print(f"  Solver: {solver_name}")
    _print(f"  Bundles / scenarios: {list(sp.bundles)}")
    _print("  Building and solving extensive form ...")

    results = None
    try:
        if hasattr(ef_solver, "solve_and_return_EF"):
            results = ef_solver.solve_and_return_EF(sp)
        elif hasattr(ef_solver, "solve"):
            results = ef_solver.solve(sp)
        else:
            _print("  Falling back to sp.create_EF + sp.solve")
            M = sp.create_EF(compact_repn=True)
            results = sp.solve(M, solver_options=solver_options)
    except Exception as e:
        _print(f"  EF solve raised: {e}")
        raise

    # Defensive objective extraction (same multi-path logic as the EGRET script)
    obj_value = None
    extraction_errors = []

    def _try_to_dict(obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if isinstance(obj, dict):
            return obj
        return None

    try:
        rd = _try_to_dict(results)
        if rd is not None and "solutions" in rd and rd["solutions"]:
            soln = next(iter(rd["solutions"].values()))
            if isinstance(soln, dict) and soln.get("objectives"):
                obj_value = float(soln["objectives"][0]["value"])
            elif hasattr(soln, "objectives") and soln.objectives:
                obj_value = float(soln.objectives[0].value)
    except Exception as e:
        extraction_errors.append(f"pathA: {e}")

    if obj_value is None:
        try:
            if hasattr(results, "obj_value") and results.obj_value is not None:
                obj_value = float(results.obj_value)
        except Exception as e:
            extraction_errors.append(f"pathB: {e}")

    if obj_value is None:
        try:
            if hasattr(results, "solutions"):
                sols = results.solutions
                rd = _try_to_dict(sols) or (sols if isinstance(sols, dict) else None)
                if rd is not None:
                    container = rd.get("solutions", rd)
                    if container:
                        soln = next(iter(container.values()))
                        if isinstance(soln, dict) and soln.get("objectives"):
                            obj_value = float(soln["objectives"][0]["value"])
        except Exception as e:
            extraction_errors.append(f"pathC: {e}")

    if obj_value is None:
        try:
            if hasattr(sp, "get_objective_value"):
                obj_value = float(sp.get_objective_value())
        except Exception as e:
            extraction_errors.append(f"pathD: {e}")

    if obj_value is None:
        _print("  Could not recover a numeric objective. Dumping results structure:")
        _print(f"  type(results) = {type(results)}")
        _print(f"  dir(results)  = {[a for a in dir(results) if not a.startswith('_')]}")
        if extraction_errors:
            for err in extraction_errors:
                _print(f"    {err}")
        raise RuntimeError(
            "Stage 6 FAILED – could not recover a numeric objective value from "
            "the extensive-form solve.  See the structure dump above."
        )

    _print(f"  EF objective value: {obj_value:.6g}")

    try:
        b0 = next(iter(sp.bundles))
        if hasattr(sp, "int_to_FirstStageVar") and b0 in sp.int_to_FirstStageVar:
            fs_map = sp.int_to_FirstStageVar[b0]
            _print("  Sample first-stage values (post-EF):")
            for i, v in list(fs_map.items())[:8]:
                try:
                    _print(f"    {v.name} = {pyo.value(v)}")
                except Exception:
                    _print(f"    {v.name} = <unavailable>")
    except Exception as e:
        _print(f"  (Could not print first-stage values: {e})")

    _print("  Stage 6 PASSED – reference objective recorded")
    return {
        "objective": obj_value,
        "results": results,
        "solver": solver_name,
    }


# ---------------------------------------------------------------------------
# Stage 7 – Peek transforms + residual discrete gate
# ---------------------------------------------------------------------------

def stage7_peek_transforms(sp, max_print: int = 20):
    """
    Build master and subproblem once via the same transform path Benders will use.

    Residual Binary / Integer count in the subproblem must be ~0 for classical dual
    cuts.  We pass additional_transforms=[relax_second_stage] explicitly — the same
    contract Stage 8 / BendersSolver.solve uses via subproblem_transforms.
    """
    _print("\n=== STAGE 7: Peek at SPAROW transforms + residual discrete gate ===")
    from sparow.benders import BendersSolver
    from sparow.sp.util import relax_second_stage

    b0 = next(iter(sp.bundles))
    _print(f"  Using bundle {b0}")
    _print("  Subproblem additional_transforms: [relax_second_stage]")

    residual_bin = None
    residual_int = None
    try:
        sub = BendersSolver._transform_to_subproblem_model(
            sp,
            b0,
            default_domain=pyo.Reals,
            remove_first_stage_only_cons=False,
            weight_obj_by_prob=True,
            remove_first_stage_objective_terms=True,
            additional_transforms=[relax_second_stage],
        )
        _print(f"  Subproblem model created: {type(sub)}")
        objs = list(sub.component_data_objects(pyo.Objective, active=True))
        _print(f"  Subproblem active objectives: {[o.name for o in objs]}")

        residual_bin = 0
        residual_int = 0
        for v in sub.component_data_objects(pyo.Var, active=True, descend_into=True):
            dom = v.domain
            if dom is pyo.Binary or (
                hasattr(dom, "name") and "binary" in str(dom).lower()
            ):
                residual_bin += 1
            elif dom is pyo.Integers or (
                hasattr(dom, "name") and "integer" in str(dom).lower()
            ):
                residual_int += 1
        _print(f"  Subproblem residual Binary vars:  {residual_bin}")
        _print(f"  Subproblem residual Integer vars: {residual_int}")

        if residual_bin > 0 or residual_int > 0:
            _print(
                "  WARNING: residual discrete variables remain after "
                "additional_transforms=[relax_second_stage].\n"
                "  Classical dual cuts require an LP subproblem.\n"
                "  Inspect residual names and whether relax_second_stage saw the "
                "post-bigM reformulation binaries."
            )
        else:
            _print(
                "  Residual discrete count = 0  → classical dual cuts are admissible"
            )
    except Exception as e:
        _print(f"  Subproblem transform raised (inspect carefully): {e}")

    # Master: do NOT pass residual relaxation (first-stage must stay discrete)
    eta_bounds = {b: (-1e5, None) for b in sp.bundles}
    try:
        sp_upper = copy.deepcopy(sp)
        master = BendersSolver._transform_to_master_model(
            sp=sp_upper,
            b=b0,
            eta_bounds_map=eta_bounds,
            lower_bounding_otherwise_enforced=False,
            fix_second_stage_vars=True,
            additional_transforms=None,
        )
        _print(f"  Master model created: {type(master)}")
        if hasattr(master, "etas"):
            _print(f"  Master etas: {list(master.etas.keys())}")
        objs = list(master.component_data_objects(pyo.Objective, active=True))
        _print(f"  Master active objectives: {[o.name for o in objs]}")
    except Exception as e:
        _print(f"  Master transform raised (inspect carefully): {e}")

    _print("  Stage 7 finished (manual inspection of any warnings above)")
    return {
        "residual_binary": residual_bin,
        "residual_integer": residual_int,
    }


# ---------------------------------------------------------------------------
# Stage 8 – Benders + EF comparison
# ---------------------------------------------------------------------------

def stage8_run_benders(
    sp,
    ef_reference: dict,
    max_iterations: int = 50,
    eta_lower: float = 0.0,
    obj_tol: float = 1e-3,
    rel_tol: float = 1e-4,
):
    """
    Run SPAROW BendersSolver and compare against the EF reference from Stage 6.

    Residual second-stage discretes are handled by the public SPAROW transform
    via the new BendersSolver.solve(subproblem_transforms=...) hook::

        subproblem_transforms=[relax_second_stage]

    Master does NOT receive residual relaxation (first-stage stay discrete).

    Other practices carried from the EGRET ladder, adapted for GTEP:
    - finite eta lower bounds
    - seed / unfix any stage-1 indicators that were fixed from in_service
    - persistent solver for master and subproblem
    - allow_infeasible_subproblems=True
    """
    _print("\n=== STAGE 8: SPAROW BendersSolver (+ EF comparison) ===")
    from sparow.benders import BendersSolver
    from sparow.sp.util import relax_second_stage

    ef_obj = ef_reference["objective"]
    _print(f"  EF reference objective: {ef_obj:.6g}")

    if eta_lower is None:
        eta_lower = 0.0
    eta_bounds_map = {b: (float(eta_lower), None) for b in sp.bundles}
    _print(f"  eta_bounds_map: {eta_bounds_map}")
    for b, bounds in eta_bounds_map.items():
        if bounds[0] is None:
            raise RuntimeError(
                f"Stage 8 FAILED – eta lower bound for bundle {b!r} is None. "
                "Provide a finite default (e.g. 0.0 or -1e6)."
            )

    # Seed / unfix first-stage variables (same rationale as EGRET)
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

    # Optional iteration banners (same idea as EGRET Stage 8)
    try:
        from or_topas.benders.benders_serial import Benders_Serial as _BendersCutGen
    except ImportError:
        try:
            from or_topas.benders import Benders_Serial as _BendersCutGen
        except ImportError:
            _BendersCutGen = None

    _iter_state = {"n": 0, "cuts_total": 0, "last_cuts": 0}
    _orig_generate_cut = None
    if _BendersCutGen is not None and hasattr(_BendersCutGen, "generate_cut"):
        _orig_generate_cut = _BendersCutGen.generate_cut

        def _generate_cut_with_progress(self, *args, **kwargs):
            cuts = _orig_generate_cut(self, *args, **kwargs)
            _iter_state["n"] += 1
            n_cuts = len(cuts) if cuts is not None else 0
            _iter_state["last_cuts"] = n_cuts
            _iter_state["cuts_total"] += n_cuts
            master_obj = None
            eta_vals = None
            try:
                m = getattr(self, "model", None) or getattr(self, "_model", None)
                if m is not None and hasattr(m, "obj"):
                    master_obj = pyo.value(m.obj)
                if m is not None and hasattr(m, "etas"):
                    eta_vals = {k: pyo.value(v) for k, v in m.etas.items()}
            except Exception:
                pass
            parts = [
                f"  --- Benders iteration {_iter_state['n']}",
                f"cuts_added={n_cuts}",
                f"cuts_total={_iter_state['cuts_total']}",
            ]
            if master_obj is not None:
                parts.append(f"master_obj={master_obj:.6g}")
            if eta_vals is not None:
                eta_str = ", ".join(
                    f"{k}:{v:.4g}" for k, v in list(eta_vals.items())[:4]
                )
                parts.append(f"eta=[{eta_str}]")
            if n_cuts == 0:
                parts.append("CONVERGED (no cuts)")
            parts.append("---")
            _print("  ".join(parts))
            return cuts

        _BendersCutGen.generate_cut = _generate_cut_with_progress

    _print(
        "  Calling BendersSolver.solve(\n"
        "      subproblem_transforms=[relax_second_stage],\n"
        "      master_transforms=None,\n"
        "      allow_infeasible_subproblems=True,\n"
        "      persistent solvers ...\n"
        "  )"
    )
    try:
        results = solver.solve(
            sp,
            eta_bounds_map,
            subproblem_transforms=[relax_second_stage],
            master_transforms=None,
        )
    finally:
        if _orig_generate_cut is not None:
            _BendersCutGen.generate_cut = _orig_generate_cut

    _print(
        f"  Solve returned after {_iter_state['n']} Benders iteration(s), "
        f"{_iter_state['cuts_total']} cut(s) total."
    )

    benders_obj = None
    if hasattr(results, "metadata") and hasattr(results, "to_dict"):
        try:
            d = results.to_dict()
            soln = next(iter(d["solutions"].values()))
            benders_obj = float(soln["objectives"][0]["value"])
        except Exception:
            pass
    if benders_obj is None and hasattr(results, "obj_value"):
        benders_obj = float(results.obj_value)

    if benders_obj is None:
        raise RuntimeError(
            "Stage 8 FAILED – could not recover a numeric objective from "
            "Benders results.  Inspect the results object."
        )

    _print(f"  Benders objective value: {benders_obj:.6g}")
    abs_diff = abs(benders_obj - ef_obj)
    tol = max(obj_tol, rel_tol * abs(ef_obj))
    _print(f"  |Benders - EF| = {abs_diff:.6g}  (tol = {tol:.6g})")

    if abs_diff > tol:
        raise RuntimeError(
            f"Stage 8 FAILED – Benders objective {benders_obj:.6g} differs from "
            f"EF reference {ef_obj:.6g} by more than tolerance {tol:.6g}. "
            "Because the EF solved successfully, the discrepancy is almost "
            "certainly in the Benders path (transforms, cut generation, "
            "probability weighting, first-stage cost handling, or eta bounds)."
        )

    _print("  Stage 8 PASSED – Benders matches EF within tolerance")
    return {
        "benders_objective": benders_obj,
        "ef_objective": ef_obj,
        "abs_diff": abs_diff,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_staged_diagnostics(
    *,
    force_minimal: bool = True,
    first_stage_names: Optional[Sequence[str]] = None,
    scenario_scales: Optional[Dict[Any, float]] = None,
    stop_after_ef: bool = True,
    run_benders: bool = False,
    ef_solver: str = "gurobi",
    ef_time_limit: float = 600,
    max_benders_iterations: int = 50,
    # size knobs (overridden by force_minimal)
    num_stages: int = 1,
    num_rep_days: int = 1,
    len_rep_days: int = 4,
    num_commit_p: int = 2,
    num_disp: int = 1,
    alpha: float = 1.0,
    flow_model: str = "CP",
    include_commitment: bool = True,
    skip_trivial_sp: bool = False,
):
    """
    Execute the staged path for GTEP under SPAROW classical Benders.

    Recommended first call after any model change::

        run_staged_diagnostics(
            force_minimal=True,
            stop_after_ef=True,
            run_benders=False,
        )

    Only set run_benders=True after Stage 6 has produced a clean EF objective
    and Stage 7 reports residual Binary ≈ 0.
    """
    stage0_check_environment()

    model, size_cfg = stage1_build_gtep_model(
        force_minimal=force_minimal,
        num_stages=num_stages,
        num_rep_days=num_rep_days,
        len_rep_days=len_rep_days,
        num_commit_p=num_commit_p,
        num_disp=num_disp,
        alpha=alpha,
        flow_model=flow_model,
        include_commitment=include_commitment,
    )

    binary_comps, integer_comps, suggested = stage2_inspect_variables(model)

    if first_stage_names is None:
        first_stage_names = list(DEFAULT_FIRST_STAGE_NAMES)
        _print(f"\n  Using DEFAULT_FIRST_STAGE_NAMES ({len(first_stage_names)} patterns)")

    stage3_verify_first_stage_names(model, first_stage_names)

    if not skip_trivial_sp:
        stage4_trivial_sparow_sp()
    else:
        _print("\n=== STAGE 4: skipped (skip_trivial_sp=True) ===")

    if scenario_scales is None:
        scenario_scales = {"single": 1.0}

    sp = stage5_gtep_sparow_sp(
        size_cfg,
        first_stage_names=first_stage_names,
        scenario_scales=scenario_scales,
        register_relax_second_stage=True,
    )

    # Critical control point
    ef_ref = stage6_solve_extensive_form(
        sp,
        solver_name=ef_solver,
        time_limit=ef_time_limit,
    )

    if stop_after_ef and not run_benders:
        _print("\n*** Staged diagnostics complete through EF (Stage 6). ***")
        _print(f"*** EF reference objective = {ef_ref['objective']:.6g} ***")
        _print("*** Re-run with run_benders=True only after this stage is clean. ***")
        _print("*** Also inspect Stage 7 residual Binary count before Benders. ***")
        return {"sp": sp, "ef_reference": ef_ref, "size_cfg": size_cfg}

    residual_info = stage7_peek_transforms(sp)

    if not run_benders:
        _print("\n*** EF + transform peek done. Set run_benders=True for Stage 8. ***")
        return {
            "sp": sp,
            "ef_reference": ef_ref,
            "size_cfg": size_cfg,
            "residual": residual_info,
        }

    # Hard gate: residual discrete should be ~0 after additional_transforms
    rb = residual_info.get("residual_binary")
    if rb is not None and rb > 0:
        _print(
            "\n*** WARNING: residual Binary > 0 after "
            "additional_transforms=[relax_second_stage]. ***\n"
            "*** Stage 8 will still run with the same transform list, ***\n"
            "*** but dual cuts may be invalid if the subproblem is not LP. ***\n"
            "*** Inspect Stage 7 residual names before trusting Stage 8. ***"
        )

    benders_out = stage8_run_benders(
        sp,
        ef_reference=ef_ref,
        max_iterations=max_benders_iterations,
    )
    return {
        "sp": sp,
        "ef_reference": ef_ref,
        "size_cfg": size_cfg,
        "residual": residual_info,
        "benders": benders_out,
    }


if __name__ == "__main__":
    # Default: Stages 0–6 only on a minimal instance.
    # Inspect the printed tables (especially Stage 2 inventory, Stage 5 map size,
    # and the EF objective) before re-running with run_benders=True.
    run_staged_diagnostics(
        force_minimal=True,
        stop_after_ef=True,
        run_benders=False,
        ef_solver="gurobi",
        ef_time_limit=300,
    )
