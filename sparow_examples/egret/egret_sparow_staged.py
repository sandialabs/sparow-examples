"""
EGRET + SPAROW – Staged Verification Path
=========================================

Goal: reach a working SPAROW Benders solve of an EGRET unit-commitment
model *only after* each prerequisite has been shown to work, with clear
error signatures when something is wrong.

Do not jump to BendersSolver.solve until the later stages pass.

Stage map
---------
0. Environment & imports
1. Load a concrete EGRET test instance and build a UC model
2. Inspect the model: sets, objective, all binary/integer variables
3. Decide and verify a first-stage variable name list
4. Build a minimal SPAROW StochasticProgram with a *trivial* builder
   (newsvendor-style) to confirm SPAROW plumbing
5. Build a SPAROW SP whose model_builder calls EGRET; inspect
   int_to_FirstStageVar, bundles, etc.
6. **Solve the extensive form** – reference objective and proof that the
   SP is a correctly formed MIP.  This is the critical control point.
7. (Optional) Peek at SPAROW master/subproblem transforms
8. Benders solve + automatic comparison against the EF objective
   (must match within tolerance)

Each stage function returns inspectable objects and prints diagnostics.
Failures raise with stage-specific messages.

Do not attempt Stage 8 until Stage 6 has produced a finite EF objective.
"""

from __future__ import annotations

import copy
import logging
import pprint
import sys
from typing import Any, Dict, List, Optional, Sequence

import pyomo.environ as pyo

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def _print(*args, **kwargs):
    """print with forced flush so stage banners appear immediately."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()


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
        import egret
        from egret.data.model_data import ModelData
        from egret.models.unit_commitment import create_tight_unit_commitment_model
        print(f"  EGRET OK  (egret at {egret.__file__})")
    except ImportError as e:
        missing.append(f"egret ({e})")

    try:
        import sparow
        from sparow.sp import stochastic_program
        print(f"  SPAROW OK (sparow at {sparow.__file__})")
    except ImportError as e:
        missing.append(f"sparow ({e})")

    try:
        import or_topas
        print(f"  OR-Topas OK")
    except ImportError as e:
        print(f"  OR-Topas not found (optional for pure SPAROW path): {e}")

    if missing:
        raise ImportError(
            "Stage 0 FAILED – missing packages:\n  " + "\n  ".join(missing)
        )
    _print("  Stage 0 PASSED")
    return True


# ---------------------------------------------------------------------------
# Stage 1 – Load EGRET instance and build a model
# ---------------------------------------------------------------------------

def _make_synthetic_tiny_uc(n_periods: int = 4) -> "ModelData":
    """
    Minimal ModelData for validating the Benders path quickly.

    Design goals (not a realistic UC):
    - 2 thermal units, 1 bus, copperplate-friendly
    - short horizon (default 4 periods)
    - min_up = min_down = 1 (no long commitment chains)
    - simple piecewise costs, single-segment startup
    - no reserves / regulation / contingencies

    Expected first-stage size with UnitOn only: 2 * n_periods binaries.
    Should converge in a handful of Benders iterations if the path is correct.
    """
    from egret.data.model_data import ModelData

    T = max(2, int(n_periods))
    # Mildly varying load that both units can cover together
    base_loads = [70.0, 95.0, 110.0, 80.0, 100.0, 90.0]
    load_vals = [base_loads[t % len(base_loads)] for t in range(T)]
    md_dict = {
        "system": {
            "time_keys": list(range(1, T + 1)),  # EGRET often uses 1-based keys
            "time_period_length_minutes": 60,
            "baseMVA": 100,
            "load_mismatch_cost": 1e4,
            "reserve_shortfall_cost": 1e4,
        },
        "elements": {
            "bus": {
                "Bus1": {"base_kv": 230},
            },
            "load": {
                "Load1": {
                    "bus": "Bus1",
                    "in_service": True,
                    "p_load": {
                        "data_type": "time_series",
                        "values": load_vals,
                    },
                },
            },
            "generator": {
                "G1": {
                    "generator_type": "thermal",
                    "bus": "Bus1",
                    "fuel": "gas",
                    "in_service": True,
                    "p_min": 20,
                    "p_max": 100,
                    "ramp_up_60min": 100,
                    "ramp_down_60min": 100,
                    "min_up_time": 1,
                    "min_down_time": 1,
                    "initial_status": 1,
                    "initial_p_output": 50,
                    "startup_cost": [[1, 50]],
                    "shutdown_cost": 0.0,
                    # Match official EGRET test format (polynomial → piecewise approx)
                    "p_cost": {
                        "data_type": "cost_curve",
                        "cost_curve_type": "polynomial",
                        "values": {"0": 100, "1": 15.0, "2": 0.001},
                    },
                },
                "G2": {
                    "generator_type": "thermal",
                    "bus": "Bus1",
                    "fuel": "gas",
                    "in_service": True,
                    "p_min": 10,
                    "p_max": 60,
                    "ramp_up_60min": 60,
                    "ramp_down_60min": 60,
                    "min_up_time": 1,
                    "min_down_time": 1,
                    "initial_status": -1,
                    "initial_p_output": 0,
                    "startup_cost": [[1, 30]],
                    "shutdown_cost": 0.0,
                    "p_cost": {
                        "data_type": "cost_curve",
                        "cost_curve_type": "polynomial",
                        "values": {"0": 50, "1": 20.0, "2": 0.002},
                    },
                },
            },
        },
    }
    return ModelData(md_dict)


def _truncate_model_data_horizon(md: "ModelData", n_periods: int) -> "ModelData":
    """
    Return a deep copy of ModelData restricted to the first n_periods time keys.
    Use this to shrink official instances (e.g. tiny_uc_tc 24h -> 4h) for
    faster Benders validation.
    """
    import copy

    md = copy.deepcopy(md)
    keys = list(md.data["system"]["time_keys"])[:n_periods]
    md.data["system"]["time_keys"] = keys
    n = len(keys)

    def _trim_ts(obj):
        if not isinstance(obj, dict):
            return
        for k, v in list(obj.items()):
            if isinstance(v, dict) and v.get("data_type") == "time_series":
                vals = v.get("values")
                if isinstance(vals, list) and len(vals) > n:
                    v["values"] = vals[:n]
            elif isinstance(v, dict):
                _trim_ts(v)

    for etype, elements in md.data.get("elements", {}).items():
        for ename, edata in elements.items():
            _trim_ts(edata)
    return md


def _locate_egret_test_json(instance_name: str = "tiny_uc_tc"):
    """
    Search several locations for an official EGRET UC test JSON.
    Returns a path string or None.
    """
    import os
    import egret

    candidates = []
    pkg_dir = os.path.dirname(egret.__file__)
    candidates.append(
        os.path.join(pkg_dir, "models", "tests", "uc_test_instances", f"{instance_name}.json")
    )
    # Common relative locations if the user has a source checkout nearby
    for rel in (
        f"egret/models/tests/uc_test_instances/{instance_name}.json",
        f"../Egret/egret/models/tests/uc_test_instances/{instance_name}.json",
        f"../egret/egret/models/tests/uc_test_instances/{instance_name}.json",
    ):
        candidates.append(os.path.abspath(rel))

    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _fetch_egret_test_json_from_github(instance_name: str = "tiny_uc_tc", dest_dir: str = None):
    """
    Download the official test JSON from the EGRET GitHub repo into dest_dir
    (default: /tmp or the current working directory).  Returns the local path
    or None on failure.
    """
    import os
    import urllib.request

    url = (
        "https://raw.githubusercontent.com/grid-parity-exchange/Egret/main/"
        f"egret/models/tests/uc_test_instances/{instance_name}.json"
    )
    if dest_dir is None:
        dest_dir = os.environ.get("TMPDIR", "/tmp")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"{instance_name}.json")
    try:
        print(f"  Fetching {url}")
        urllib.request.urlretrieve(url, dest)
        if os.path.isfile(dest) and os.path.getsize(dest) > 100:
            return dest
    except Exception as e:
        print(f"  GitHub fetch failed: {e}")
    return None


def stage1_build_egret_model(
    instance_name: str = "tiny_uc_tc",
    explicit_path: str = None,
    allow_synthetic: bool = True,
    allow_github_fetch: bool = True,
    force_synthetic: bool = False,
    n_periods: Optional[int] = None,
):
    """
    Load an EGRET UC test instance and create a ConcreteModel.

    Resolution order
    ----------------
    0. force_synthetic=True → always use the minimal synthetic ModelData
    1. explicit_path (if given)
    2. package / source-tree locations via _locate_egret_test_json
    3. download from GitHub raw (if allow_github_fetch)
    4. synthetic minimal ModelData (if allow_synthetic)

    n_periods : if set, truncate the time horizon to the first n periods
    (useful to shrink tiny_uc_tc from 24h to 4h for faster Benders checks).

    Success signature
    -----------------
    - Returns (ModelData, ConcreteModel)
    - model has at least one active Objective
    """
    _print(f"\n=== STAGE 1: Build EGRET model ({instance_name}) ===")
    from egret.data.model_data import ModelData
    from egret.models.unit_commitment import create_tight_unit_commitment_model
    import os

    md = None
    source = None

    # 0. forced synthetic (preferred for quick Benders validation)
    if force_synthetic:
        T = n_periods if n_periods is not None else 4
        _print(f"  force_synthetic=True → 2-gen copperplate UC, T={T}")
        md = _make_synthetic_tiny_uc(n_periods=T)
        source = f"synthetic(T={T})"
        n_periods = None  # already sized

    # 1. explicit path
    if md is None and explicit_path is not None:
        _print(f"  Trying explicit path: {explicit_path}")
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(
                f"Stage 1 FAILED – explicit_path does not exist: {explicit_path}"
            )
        md = ModelData.read(explicit_path)
        source = explicit_path

    # 2. package / nearby source tree
    if md is None:
        path = _locate_egret_test_json(instance_name)
        if path is not None:
            _print(f"  Reading {path}")
            md = ModelData.read(path)
            source = path
        else:
            _print(
                "  Official test JSON not found next to the installed egret package.\n"
                "  (This is normal for conda/pip installs – models/tests/ is often omitted.)"
            )

    # 3. GitHub download
    if md is None and allow_github_fetch:
        path = _fetch_egret_test_json_from_github(instance_name)
        if path is not None:
            _print(f"  Reading downloaded file {path}")
            md = ModelData.read(path)
            source = f"github:{path}"

    # 4. synthetic fallback
    if md is None:
        if not allow_synthetic:
            raise FileNotFoundError(
                "Stage 1 FAILED – could not locate or download a test instance, "
                "and allow_synthetic=False.  Pass explicit_path= to a local JSON "
                "or set allow_synthetic=True."
            )
        T = n_periods if n_periods is not None else 4
        _print(f"  Falling back to synthetic ModelData (2 gens, T={T}).")
        md = _make_synthetic_tiny_uc(n_periods=T)
        source = f"synthetic(T={T})"
        n_periods = None

    # Optional horizon truncation of a loaded official instance
    if n_periods is not None:
        _print(f"  Truncating time horizon to first {n_periods} periods")
        md = _truncate_model_data_horizon(md, n_periods)
        source = f"{source}|T={n_periods}"

    print(f"  ModelData source: {source}")
    print(f"  system keys: {list(md.data.get('system', {}).keys())}")
    print(f"  element types: {list(md.data.get('elements', {}).keys())}")

    # copperplate avoids uncopyable PTDF objects (_PTDFs, VirtualPTDFMatrix)
    # that break SPAROW deepcopy / master transforms. Revisit ptdf later.
    network = "copperplate_power_flow"
    model = create_tight_unit_commitment_model(
        md, network_constraints=network, relaxed=False
    )
    _print(f"  network_constraints: {network}")
    _print(f"  Model type: {type(model)}")
    _print(f"  Model name: {getattr(model, 'name', None)}")

    objs = list(model.component_data_objects(pyo.Objective, active=True, descend_into=True))
    if not objs:
        raise RuntimeError(
            "Stage 1 FAILED – EGRET model has no active Objective. "
            "SPAROW cost splitting and Benders will not work."
        )
    _print(f"  Active objectives: {[o.name for o in objs]}")

    for sname in ("ThermalGenerators", "TimePeriods", "Buses"):
        if hasattr(model, sname):
            s = getattr(model, sname)
            print(f"  Set {sname}: length {len(s)}")
        else:
            print(f"  Set {sname}: NOT PRESENT")

    _print("  Stage 1 PASSED")
    return md, model


# ---------------------------------------------------------------------------
# Stage 2 – Inspect binaries / candidate first-stage variables
# ---------------------------------------------------------------------------

def stage2_inspect_variables(model: pyo.ConcreteModel):
    """
    Print every binary/integer variable component and a sample of indices.

    This is the critical inspection point for choosing first_stage_variables.

    Success signature
    -----------------
    - At least one binary component whose name is a plausible commitment
      indicator (UnitOn, UnitStart, UnitStop, ...).
    - Printed table of (component name, domain, #indices, sample indices).

    Failure signatures
    ------------------
    - No binary variables at all → wrong model or relaxed=True was used.
    - Only continuous variables → you built an ED, not a UC.
    - Unexpected names → formulation uses a non-standard status representation;
      you must adjust the first_stage_variables list accordingly.
    """
    _print("\n=== STAGE 2: Inspect binary / integer variables ===")

    binary_comps = []
    integer_comps = []
    continuous_sample = []

    for comp in model.component_objects(pyo.Var, descend_into=True):
        # Peek at domain of a representative element
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
            if len(continuous_sample) < 5:
                continuous_sample.append(info)

    print("  Binary components:")
    if not binary_comps:
        print("    *** NONE FOUND ***")
        raise RuntimeError(
            "Stage 2 FAILED – no Binary variables. "
            "Did you pass relaxed=True, or build an economic-dispatch model?"
        )
    for b in binary_comps:
        print(f"    {b['name']:40s}  n={b['n']:5d}  sample_idx={b['sample_indices']}")

    print("  Integer (non-binary) components:")
    if not integer_comps:
        print("    (none)")
    for b in integer_comps:
        print(f"    {b['name']:40s}  n={b['n']:5d}  sample_idx={b['sample_indices']}")

    print("  Continuous sample (first few):")
    for b in continuous_sample:
        print(f"    {b['name']:40s}  n={b['n']:5d}")

    # Suggested first-stage list.  Include regulation binaries so the Benders
    # subproblem has no residual Binary vars (classical L-shaped needs an LP).
    keywords = (
        "uniton", "unitstart", "unitstop",
        "startup", "shutdown", "status",
        "regulation",  # RegulationOn
    )
    suggested = [
        b["name"] for b in binary_comps
        if any(k in b["name"].lower() for k in keywords)
    ]
    _print(f"\n  Suggested first_stage_variables (heuristic): {suggested}")
    _print("  Stage 2 PASSED – inspect the list above before continuing")
    return binary_comps, integer_comps, suggested


# ---------------------------------------------------------------------------
# Stage 3 – Verify a concrete first-stage name list against the model
# ---------------------------------------------------------------------------

def stage3_verify_first_stage_names(
    model: pyo.ConcreteModel,
    first_stage_names: Sequence[str],
):
    """
    Confirm that every name in first_stage_names exists on the model and
    yields at least one variable.

    Success signature
    -----------------
    - For each name: component found, number of variables reported.
    - Returns the flattened list of VarData objects.

    Failure signatures
    ------------------
    - AttributeError / "not found" → name does not exist on this formulation.
      Fix: re-run Stage 2 and correct the list.
    - Component exists but is continuous → you picked a continuous var by mistake.
    - Empty index set → data problem (no generators / time periods).
    """
    _print("\n=== STAGE 3: Verify first-stage name list ===")
    print(f"  Candidate names: {list(first_stage_names)}")

    found_vars = []
    for name in first_stage_names:
        if not hasattr(model, name):
            raise AttributeError(
                f"Stage 3 FAILED – model has no component '{name}'. "
                "Re-run Stage 2 and choose names that actually exist."
            )
        comp = getattr(model, name)
        if not isinstance(comp, pyo.Var):
            raise TypeError(
                f"Stage 3 FAILED – '{name}' exists but is not a Var "
                f"(type={type(comp)})."
            )
        # Domain check on a sample
        sample = next(comp.values()) if comp.is_indexed() else comp
        if sample.domain is not pyo.Binary and "binary" not in str(sample.domain).lower():
            print(f"  WARNING: '{name}' sample domain is {sample.domain} (not Binary)")

        n = len(comp) if comp.is_indexed() else 1
        print(f"  '{name}': found, n={n}, domain={sample.domain}")
        if comp.is_indexed():
            found_vars.extend(list(comp.values()))
        else:
            found_vars.append(comp)

    if not found_vars:
        raise RuntimeError("Stage 3 FAILED – first-stage name list produced zero variables")

    print(f"  Total first-stage VarData objects: {len(found_vars)}")
    _print("  Stage 3 PASSED")
    return found_vars


# ---------------------------------------------------------------------------
# Stage 4 – Minimal SPAROW SP with a trivial (non-EGRET) builder
# ---------------------------------------------------------------------------

def stage4_trivial_sparow_sp():
    """
    Build a SPAROW StochasticProgram using the official newsvendor pattern.
    Confirms that SPAROW itself is healthy before we involve EGRET.

    Success signature
    -----------------
    - sp.bundles is non-empty
    - after model construction, int_to_FirstStageVar maps exist
    - a simple solve or just model creation succeeds

    Failure signatures
    ------------------
    - ImportError / AttributeError on stochastic_program → SPAROW version mismatch
    - bundles is None or empty after initialize_model → bundling scheme problem
    - first-stage map empty → name matching failed even on the trivial model
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
    # Do NOT pass name=...  Leave it at the default (None).
    # This is the pattern the Benders path expects; using an explicit
    # name like "news" here would train us on a shape that later breaks
    # BendersSolver.
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

    # Force creation of one bundle EF so that cuid maps are populated
    b0 = next(iter(sp.bundles))
    try:
        m = sp.create_bundle_EF(b=b0, w=None, x_bar=None, rho=None, cached=False)
        print(f"  create_bundle_EF succeeded for bundle {b0}")
    except Exception as e:
        raise RuntimeError(
            f"Stage 4 FAILED – create_bundle_EF raised: {e}"
        ) from e

    # Inspect first-stage map
    if hasattr(sp, "int_to_FirstStageVar") and b0 in sp.int_to_FirstStageVar:
        fs_map = sp.int_to_FirstStageVar[b0]
        _print(f"  int_to_FirstStageVar[{b0}]: {len(fs_map)} entries")
        for i, v in list(fs_map.items())[:3]:
            print(f"    {i}: {v.name} = {v}")
    else:
        print("  WARNING: int_to_FirstStageVar not yet populated (may be lazy)")

    _print("  Stage 4 PASSED")
    return sp


# ---------------------------------------------------------------------------
# Stage 5 – SPAROW SP whose builder calls EGRET
# ---------------------------------------------------------------------------

def stage5_egret_sparow_sp(
    base_md,
    first_stage_names: Sequence[str],
    scenario_scales: Optional[Dict[Any, float]] = None,
    formulation: str = "tight",
):
    """
    Construct a SPAROW SP that uses an EGRET model_builder.
    Do *not* call Benders yet – only inspect the resulting maps and models.

    Success signature
    -----------------
    - bundles non-empty
    - int_to_FirstStageVar non-empty for each bundle
    - the variables in the map have names that match first_stage_names
    - create_bundle_EF returns a model that still contains those variables

    Failure signatures
    ------------------
    - int_to_FirstStageVar empty → SPAROW could not find any variable whose
      ComponentUID matched the names you supplied.  Go back to Stage 2/3.
    - create_bundle_EF raises inside EGRET builder → data / formulation problem
    - map contains continuous variables → wrong names were chosen
    """
    _print("\n=== STAGE 5: SPAROW SP with EGRET model_builder ===")
    from sparow.sp import stochastic_program
    from egret.models.unit_commitment import create_tight_unit_commitment_model

    if scenario_scales is None:
        scenario_scales = {"base": 1.0}

    def scale_md(md, scale, scen_id):
        md = copy.deepcopy(md)
        for load in md.data.get("elements", {}).get("load", {}).values():
            if "p_load" in load and isinstance(load["p_load"], dict) \
                    and load["p_load"].get("data_type") == "time_series":
                load["p_load"] = dict(load["p_load"])
                load["p_load"]["values"] = [v * scale for v in load["p_load"]["values"]]
        md.data.setdefault("system", {})["scenario_id"] = scen_id
        return md

    scenarios = []
    for sid, scale in scenario_scales.items():
        scenarios.append({
            "ID": sid,
            "load_scale": scale,
            "base_md": base_md,
        })

    model_data = {"data": {}, "scenarios": scenarios}

    def egret_builder(data, args=None):
        scale = float(data.get("load_scale", 1.0))
        base = data.get("base_md")
        sid = data.get("ID")
        if base is None:
            raise KeyError("egret_builder expected 'base_md' in scenario data")
        md = scale_md(base, scale, sid)
        m = create_tight_unit_commitment_model(
            md, network_constraints="copperplate_power_flow", relaxed=False
        )
        # Alias objective for SPAROW robustness
        objs = list(m.component_data_objects(pyo.Objective, active=True))
        if objs and objs[0].name not in ("o", "obj"):
            m.o = pyo.Objective(expr=objs[0].expr, sense=objs[0].sense)
            objs[0].deactivate()
        return m

    sp = stochastic_program(first_stage_variables=list(first_stage_names))
    sp.initialize_application(app_data={"formulation": formulation})
    # Do NOT pass name=...  Leave default (None). Same pattern as Stage 4
    # and as expected by BendersSolver.
    sp.initialize_model(
        model_data=model_data,
        model_builder=egret_builder,
        default=True,
    )

    _print(f"  default_model: {sp.default_model!r}")
    _print(f"  bundles: {list(sp.bundles) if sp.bundles else None}")
    if not sp.bundles:
        raise RuntimeError("Stage 5 FAILED – no bundles")

    # Force map population by building one EF
    b0 = next(iter(sp.bundles))
    print(f"  Building EF for bundle {b0} ...")
    try:
        m = sp.create_bundle_EF(b=b0, w=None, x_bar=None, rho=None, cached=False)
    except Exception as e:
        raise RuntimeError(
            f"Stage 5 FAILED – create_bundle_EF / EGRET builder raised:\n  {e}"
        ) from e

    # Inspect the first-stage map
    if not hasattr(sp, "int_to_FirstStageVar") or b0 not in sp.int_to_FirstStageVar:
        raise RuntimeError(
            "Stage 5 FAILED – int_to_FirstStageVar missing after create_bundle_EF. "
            "SPAROW did not recognise any of the first_stage_variables names "
            f"{list(first_stage_names)} inside the EGRET model. "
            "Re-run Stage 2 and correct the name list."
        )

    fs_map = sp.int_to_FirstStageVar[b0]
    _print(f"  int_to_FirstStageVar[{b0}]: {len(fs_map)} variables")
    if len(fs_map) == 0:
        raise RuntimeError(
            "Stage 5 FAILED – first-stage map is empty. "
            "The names in first_stage_variables were not found as Var components "
            "in the EGRET model.  Check Stage 2 output."
        )

    # Show a few
    for i, v in list(fs_map.items())[:5]:
        print(f"    [{i}] {v.name}  domain={v.domain}")

    # Sanity: all should be binary (or integer)
    non_bin = [v for v in fs_map.values()
               if v.domain is not pyo.Binary and "binary" not in str(v.domain).lower()]
    if non_bin:
        print(f"  WARNING: {len(non_bin)} first-stage vars are not Binary")

    _print("  Stage 5 PASSED")
    return sp


# ---------------------------------------------------------------------------
# Stage 6 – Solve the Extensive Form (reference solution)
# ---------------------------------------------------------------------------

def stage6_solve_extensive_form(
    sp,
    solver_name: str = "gurobi",
    solver_options: Optional[dict] = None,
    time_limit: Optional[float] = 300,
):
    """
    Solve the extensive form of the SPAROW SP.  This is the critical control
    point before any Benders work.

    Why this stage exists
    ---------------------
    - Confirms the SP (EGRET models + first-stage identification + probabilities)
      is a correctly formed stochastic program that a MIP solver can solve.
    - Produces a reference objective value (and optionally solution) that later
      Benders runs must match within tolerance.
    - Isolates formulation / data / mapping bugs from Benders-specific bugs
      (transforms, duals, cut generation, eta bounds, ...).

    Success signature
    -----------------
    - EF model is created (sp.create_EF or ExtensiveFormSolver).
    - Solver returns optimal (or user-accepted) termination.
    - A finite objective value is recorded and returned.
    - Printed summary of objective and a few first-stage values.

    Failure signatures
    ------------------
    - create_EF / solve raises → model construction or solver interface problem
      (often objective naming, unbounded, or missing duals not relevant here).
    - Infeasible EF → data inconsistency across scenarios or bad constraints
      coming from EGRET.
    - Unbounded EF → missing bounds or objective sense problem.
    - Objective is None → solver did not load a solution.
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
            _print("  ExtensiveFormSolver has neither solve nor solve_and_return_EF; "
                   "falling back to sp.create_EF + sp.solve")
            M = sp.create_EF(compact_repn=True)
            results = sp.solve(M, solver_options=solver_options)
    except Exception as e:
        _print(f"  EF solve raised: {e}")
        raise

    # ------------------------------------------------------------------
    # Defensive objective extraction (SPAROW result shapes vary)
    # Canonical pattern: results.to_dict()["solutions"][key]["objectives"][0]["value"]
    # ------------------------------------------------------------------
    obj_value = None
    extraction_errors = []

    def _try_to_dict(obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if isinstance(obj, dict):
            return obj
        return None

    # Path A: SparowPoolManager / solutions pool
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

    # Path B: results.obj_value
    if obj_value is None:
        try:
            if hasattr(results, "obj_value") and results.obj_value is not None:
                obj_value = float(results.obj_value)
        except Exception as e:
            extraction_errors.append(f"pathB: {e}")

    # Path C: nested .solutions attribute
    if obj_value is None:
        try:
            if hasattr(results, "solutions"):
                sols = results.solutions
                rd = _try_to_dict(sols) or (sols if isinstance(sols, dict) else None)
                if rd is not None:
                    # maybe rd is already {"solutions": {...}} or just the solutions dict
                    container = rd.get("solutions", rd)
                    if container:
                        soln = next(iter(container.values()))
                        if isinstance(soln, dict) and soln.get("objectives"):
                            obj_value = float(soln["objectives"][0]["value"])
        except Exception as e:
            extraction_errors.append(f"pathC: {e}")

    # Path D: SP helper after solve
    if obj_value is None:
        try:
            if hasattr(sp, "get_objective_value"):
                obj_value = float(sp.get_objective_value())
        except Exception as e:
            extraction_errors.append(f"pathD: {e}")

    # Path E: metadata
    if obj_value is None:
        try:
            if hasattr(results, "metadata") and hasattr(results.metadata, "get"):
                for key in ("objective", "obj_value", "objective_value"):
                    if key in results.metadata and results.metadata[key] is not None:
                        obj_value = float(results.metadata[key])
                        break
        except Exception as e:
            extraction_errors.append(f"pathE: {e}")

    if obj_value is None:
        _print("  Could not recover a numeric objective. Dumping results structure:")
        _print(f"  type(results) = {type(results)}")
        _print(f"  dir(results)  = {[a for a in dir(results) if not a.startswith('_')]}")
        if extraction_errors:
            _print("  Extraction attempts:")
            for err in extraction_errors:
                _print(f"    {err}")
        try:
            rd = _try_to_dict(results)
            if rd is not None:
                _print(f"  to_dict keys = {list(rd.keys())}")
                # print a short preview
                preview = {k: type(v).__name__ for k, v in rd.items()}
                _print(f"  to_dict value types = {preview}")
                if "solutions" in rd:
                    sols = rd["solutions"]
                    _print(f"  solutions keys = {list(sols.keys())[:5]}")
                    if sols:
                        first = next(iter(sols.values()))
                        _print(f"  first solution type = {type(first)}")
                        if isinstance(first, dict):
                            _print(f"  first solution keys = {list(first.keys())}")
        except Exception as e:
            _print(f"  to_dict dump failed: {e}")
        raise RuntimeError(
            "Stage 6 FAILED – could not recover a numeric objective value from "
            "the extensive-form solve.  See the structure dump above.  "
            "Gurobi may still have found a solution; the issue is only in how "
            "we read the SPAROW results object."
        )

    _print(f"  EF objective value: {obj_value:.6g}")

    # Sample first-stage values
    try:
        b0 = next(iter(sp.bundles))
        if hasattr(sp, "int_to_FirstStageVar") and b0 in sp.int_to_FirstStageVar:
            fs_map = sp.int_to_FirstStageVar[b0]
            _print("  Sample first-stage values (post-EF):")
            for i, v in list(fs_map.items())[:8]:
                _print(f"    {v.name} = {pyo.value(v)}")
    except Exception as e:
        _print(f"  (Could not print first-stage values: {e})")

    _print("  Stage 6 PASSED – reference objective recorded")
    return {
        "objective": obj_value,
        "results": results,
        "solver": solver_name,
    }


# ---------------------------------------------------------------------------
# Stage 7 – Peek at master / subproblem transforms (optional)
# ---------------------------------------------------------------------------

def stage7_peek_transforms(sp, max_print: int = 20):
    """
    Call SPAROW's internal transform helpers on one bundle and report what
    the master and a subproblem look like *before* any Benders solve.
    """
    _print("\n=== STAGE 7: Peek at SPAROW transforms (no solve) ===")
    from sparow.benders import BendersSolver

    b0 = next(iter(sp.bundles))
    print(f"  Using bundle {b0}")

    try:
        sub = BendersSolver._transform_to_subproblem_model(
            sp, b0,
            default_domain=pyo.Reals,
            remove_first_stage_only_cons=False,
            weight_obj_by_prob=True,
            remove_first_stage_objective_terms=True,
        )
        print(f"  Subproblem model created: {type(sub)}")
        objs = list(sub.component_data_objects(pyo.Objective, active=True))
        print(f"  Subproblem active objectives: {[o.name for o in objs]}")
        n_bin = sum(
            1 for v in sub.component_data_objects(pyo.Var, active=True)
            if v.domain is pyo.Binary or "binary" in str(v.domain).lower()
        )
        print(f"  Subproblem residual Binary vars (should be 0 or few): {n_bin}")
    except Exception as e:
        print(f"  Subproblem transform raised (inspect carefully): {e}")

    eta_bounds = {b: (-1e5, None) for b in sp.bundles}
    try:
        sp_upper = copy.deepcopy(sp)
        master = BendersSolver._transform_to_master_model(
            sp=sp_upper,
            b=b0,
            eta_bounds_map=eta_bounds,
            lower_bounding_otherwise_enforced=False,
            fix_second_stage_vars=True,
        )
        print(f"  Master model created: {type(master)}")
        if hasattr(master, "etas"):
            print(f"  Master etas: {list(master.etas.keys())}")
        objs = list(master.component_data_objects(pyo.Objective, active=True))
        print(f"  Master active objectives: {[o.name for o in objs]}")
    except Exception as e:
        print(f"  Master transform raised (inspect carefully): {e}")

    print("  Stage 7 finished (manual inspection of any warnings above)")
    return True


# ---------------------------------------------------------------------------
# Stage 8 – Benders, with mandatory comparison to EF objective
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
    Run SPAROW BendersSolver and compare the obtained objective against the
    extensive-form reference from Stage 6.

    Success signature
    -----------------
    - Benders terminates (no cuts or max iterations).
    - |benders_obj - ef_obj| <= max(obj_tol, rel_tol * |ef_obj|).

    Failure signatures
    ------------------
    - Benders crashes → transform / dual / solver issues (EF already worked,
      so the bug is Benders-specific).
    - Objective mismatch beyond tolerance → cut generation, probability
      weighting, first-stage cost handling, or eta bounds are wrong.

    Critical defaults (first Benders iteration)
    -------------------------------------------
    - eta_bounds_map must give every bundle a *finite* lower bound.  Missing
      or None lower bounds produce degenerate constraints on iteration 1.
    - Every first-stage VarData must have a numeric .value before the
      OR-Topas standard_lp transform runs.  Uninitialized (None) complicating
      variables cause expressions to collapse to a Python True and raise
      ValueError: Invalid constraint expression ... trivial Boolean (True).
    """
    _print("\n=== STAGE 8: SPAROW BendersSolver (+ EF comparison) ===")
    from sparow.benders import BendersSolver

    ef_obj = ef_reference["objective"]
    _print(f"  EF reference objective: {ef_obj:.6g}")

    # ------------------------------------------------------------------
    # eta bounds: every bundle must have an explicit finite lower bound.
    # UC second-stage costs are non-negative and on the order of the EF
    # objective (~5e5).  Default lower bound 0.0 is valid; use a large
    # negative (e.g. -1e6) only if costs can be negative.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # First-stage initialization + unfix.
    #
    # 1. Seed every complicating variable so the standard_lp transform never
    #    sees value=None (SPAROW master-side fallback: lb → ub → 0.0).
    # 2. Unfix any variables that EGRET fixed for initial conditions.
    #    Fixed vars make pure first-stage constraints evaluate to numeric
    #    constants; the transform then builds `True` and Pyomo raises
    #    ValueError.  Unfixing preserves lb/ub so the restriction remains
    #    but the expression still contains a variable.
    # ------------------------------------------------------------------
    n_seeded = 0
    n_unfixed = 0
    for b in sp.bundles:
        fs_map = getattr(sp, "int_to_FirstStageVar", {}).get(b, {})
        for v in fs_map.values():
            if v.fixed:
                # Keep the numeric restriction via bounds, drop the fixed flag
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

    # ------------------------------------------------------------------
    # Monkey-patch SPAROW's subproblem setup so pure first-stage constraints
    # are removed from the subproblem (classical Benders: those belong only
    # on the master).  Without this, EGRET logic constraints that involve
    # only UnitOn/UnitStart/UnitStop become 0 <= constant after the A x
    # split and raise the trivial-Boolean error.
    # ------------------------------------------------------------------
    _orig_setup = BendersSolver._setup_topas_subproblem

    def _setup_with_remove_fs_only_cons(
        sp_lower, b_lower, sp_upper, b_upper, remove_first_stage_objective_terms
    ):
        model_lower = BendersSolver._transform_to_subproblem_model(
            sp_lower,
            b_lower,
            default_domain=pyo.Reals,
            remove_first_stage_objective_terms=remove_first_stage_objective_terms,
            remove_first_stage_only_cons=True,  # <-- critical for EGRET UC
        )
        # Classical Benders needs an LP subproblem for duals (Pi).  Any residual
        # Binary vars (e.g. UnitStart/UnitStop when only UnitOn is first-stage)
        # make the subproblem a MIP and Gurobi cannot return duals.  Relax them
        # to continuous [0, 1].
        n_relaxed = 0
        for v in model_lower.component_data_objects(pyo.Var, active=True, descend_into=True):
            if v.domain is pyo.Binary or (
                hasattr(v.domain, "name") and "binary" in str(v.domain).lower()
            ):
                v.domain = pyo.UnitInterval  # continuous [0, 1]
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

    BendersSolver._setup_topas_subproblem = staticmethod(
        _setup_with_remove_fs_only_cons
    )

    solver = BendersSolver()
    # Master and subproblem must both be persistent when is_persistent_solver=True,
    # otherwise SPAROW calls .set_instance on a GUROBIFILE solver and crashes.
    #
    # allow_infeasible_subproblems=True is required for classical Benders:
    # the first master proposal is almost always infeasible for the dispatch
    # subproblem.  OR-Topas then generates a feasibility cut from the Farkas
    # ray.  Without this flag it raises on infeasibleOrUnbounded.
    solver.set_options(
        solver="gurobi_persistent",
        subproblem_solver="gurobi_persistent",
        max_iterations=max_iterations,
        is_persistent_solver=True,
        allow_infeasible_subproblems=True,
        loglevel="INFO",
    )
    # Propagate onto the instance in case attribute names differ across versions
    for attr in ("allow_infeasible_subproblems", "allow_infeasible"):
        if hasattr(solver, attr):
            setattr(solver, attr, True)

    # ------------------------------------------------------------------
    # Per-iteration progress banners.
    #
    # SPAROW's solve loop does not print iteration summaries at INFO level;
    # the console fills with Gurobi noise and it is hard to see that Benders
    # cuts are being generated.  Hook generate_cut on the OR-Topas serial
    # class so each completed iteration prints a clear one-line summary.
    # ------------------------------------------------------------------
    try:
        from or_topas.benders.benders_serial import Benders_Serial as _BendersCutGen
    except ImportError:
        try:
            from or_topas.benders import Benders_Serial as _BendersCutGen
        except ImportError:
            _BendersCutGen = None

    _iter_state = {
        "n": 0,
        "cuts_total": 0,
        "last_cuts": 0,
    }
    _orig_generate_cut = None
    if _BendersCutGen is not None and hasattr(_BendersCutGen, "generate_cut"):
        _orig_generate_cut = _BendersCutGen.generate_cut

        def _generate_cut_with_progress(self, *args, **kwargs):
            # Master has just been solved; generate_cut is about to (re)solve
            # subproblems and return new cuts.
            cuts = _orig_generate_cut(self, *args, **kwargs)
            _iter_state["n"] += 1
            n_cuts = len(cuts) if cuts is not None else 0
            _iter_state["last_cuts"] = n_cuts
            _iter_state["cuts_total"] += n_cuts

            # Best-effort master objective / eta values for context
            master_obj = None
            eta_vals = None
            try:
                # OR-Topas holds the master model as self.model or similar
                m = getattr(self, "model", None) or getattr(self, "_model", None)
                if m is not None and hasattr(m, "obj"):
                    master_obj = pyo.value(m.obj)
                if m is not None and hasattr(m, "etas"):
                    eta_vals = {
                        k: pyo.value(v) for k, v in m.etas.items()
                    }
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
                # keep it short
                eta_str = ", ".join(f"{k}:{v:.4g}" for k, v in list(eta_vals.items())[:4])
                parts.append(f"eta=[{eta_str}]")
            if n_cuts == 0:
                parts.append("CONVERGED (no cuts)")
            parts.append("---")
            _print("  ".join(parts))
            return cuts

        _BendersCutGen.generate_cut = _generate_cut_with_progress

    _print(
        "  Calling BendersSolver.solve "
        "(remove_first_stage_only_cons=True, allow_infeasible=True) ..."
    )
    _print(
        "  (Per-iteration banners: cuts_added / cuts_total / master_obj / eta)"
    )
    try:
        results = solver.solve(sp, eta_bounds_map)
    finally:
        # Restore original setup so later experiments are not affected
        BendersSolver._setup_topas_subproblem = _orig_setup
        if _orig_generate_cut is not None:
            _BendersCutGen.generate_cut = _orig_generate_cut

    _print(
        f"  Solve returned after {_iter_state['n']} Benders iteration(s), "
        f"{_iter_state['cuts_total']} cut(s) total."
    )

    # Recover Benders objective
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

    print(f"  Benders objective value: {benders_obj:.6g}")
    abs_diff = abs(benders_obj - ef_obj)
    rel_diff = abs_diff / max(abs(ef_obj), 1e-12)
    tol = max(obj_tol, rel_tol * abs(ef_obj))
    print(f"  |Benders - EF| = {abs_diff:.6g}  (tol = {tol:.6g})")

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
# Driver: run stages sequentially, stop at first failure
# ---------------------------------------------------------------------------

def run_staged_diagnostics(
    instance_name: str = "tiny_uc_tc",
    first_stage_names: Optional[Sequence[str]] = None,
    scenario_scales: Optional[Dict[Any, float]] = None,
    stop_after_ef: bool = True,
    run_benders: bool = False,
    ef_solver: str = "gurobi",
    ef_time_limit: float = 300,
    explicit_path: Optional[str] = None,
    allow_synthetic: bool = True,
    allow_github_fetch: bool = True,
    force_synthetic: bool = False,
    n_periods: Optional[int] = None,
    max_benders_iterations: int = 50,
):
    """
    Execute the staged path.

    For a *quick* Benders validation (recommended next step after the
    50-iteration incomplete run on full tiny_uc_tc)::

        run_staged_diagnostics(
            force_synthetic=True,
            n_periods=4,
            first_stage_names=["UnitOn"],   # lean master
            stop_after_ef=False,
            run_benders=True,
            max_benders_iterations=30,
        )

    That yields ~8 first-stage binaries and should converge in a few iterations
    if the Benders path is correct, before reintroducing the full instance.

    Parameters
    ----------
    force_synthetic : always use the 2-gen minimal ModelData (ignores instance_name)
    n_periods       : truncate horizon (or size the synthetic instance)
    max_benders_iterations : passed to Stage 8
    """
    stage0_check_environment()

    md, model = stage1_build_egret_model(
        instance_name,
        explicit_path=explicit_path,
        allow_synthetic=allow_synthetic,
        allow_github_fetch=allow_github_fetch,
        force_synthetic=force_synthetic,
        n_periods=n_periods,
    )

    binary_comps, integer_comps, suggested = stage2_inspect_variables(model)

    if first_stage_names is None:
        if force_synthetic:
            # Lean first-stage for the synthetic check: UnitOn only
            first_stage_names = [b["name"] for b in binary_comps
                                 if "uniton" in b["name"].lower()]
            if not first_stage_names:
                first_stage_names = suggested
        elif not suggested:
            raise RuntimeError(
                "No heuristic first-stage names suggested. "
                "Supply first_stage_names explicitly after inspecting Stage 2."
            )
        else:
            first_stage_names = suggested
        _print(f"\n  Using first_stage_names = {list(first_stage_names)}")

    stage3_verify_first_stage_names(model, first_stage_names)

    stage4_trivial_sparow_sp()

    if scenario_scales is None:
        scenario_scales = {"base": 1.0}

    sp = stage5_egret_sparow_sp(
        md,
        first_stage_names=first_stage_names,
        scenario_scales=scenario_scales,
    )

    # Critical control point: solve the extensive form
    ef_ref = stage6_solve_extensive_form(
        sp,
        solver_name=ef_solver,
        time_limit=ef_time_limit,
    )

    if stop_after_ef and not run_benders:
        print("\n*** Staged diagnostics complete through EF (Stage 6). ***")
        print(f"*** EF reference objective = {ef_ref['objective']:.6g} ***")
        print("*** Re-run with run_benders=True only after this stage is clean. ***")
        return {"sp": sp, "ef_reference": ef_ref}

    # Optional transform inspection
    stage7_peek_transforms(sp)

    if not run_benders:
        print("\n*** EF + transform peek done. Set run_benders=True for Stage 8. ***")
        return {"sp": sp, "ef_reference": ef_ref}

    # Benders with mandatory EF comparison
    benders_out = stage8_run_benders(
        sp,
        ef_reference=ef_ref,
        max_iterations=max_benders_iterations,
    )
    return {
        "sp": sp,
        "ef_reference": ef_ref,
        "benders": benders_out,
    }


if __name__ == "__main__":
    # Default: stages 0–6 only (environment → EGRET model → FS inspection →
    # trivial SPAROW → EGRET SPAROW SP → Extensive Form solve).
    # Inspect the printed tables, especially Stage 2 and the EF objective,
    # before re-running with run_benders=True.
    run_staged_diagnostics(
        instance_name="tiny_uc_tc",
        first_stage_names=None,          # let Stage 2 suggest
        scenario_scales={"base": 1.0},   # single scenario first
        stop_after_ef=True,
        run_benders=False,
        ef_solver="gurobi",
        ef_time_limit=120,
    )
