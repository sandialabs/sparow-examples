"""
EGRET + SPAROW – Multi-Scenario Staged Verification Path
=======================================================

Goal: reach a working SPAROW Benders solve of an EGRET unit-commitment
model under a *true multi-scenario* stochastic program, only after each
prerequisite has been shown to work, with clear error signatures when
something is wrong.

This is the multi-scenario counterpart to egret_sparow_staged.py.
The single-scenario path is assumed to work; this script forces the
features required for ≥2 scenarios with (optionally non-uniform)
probabilities.

Do not jump to BendersSolver.solve until the later stages pass.

Stage map
---------
0. Environment & imports
1. Load / build a concrete EGRET test instance (base ModelData)
2. Inspect the model: sets, objective, all binary/integer variables
3. Decide and verify a first-stage variable name list
4. Build a minimal multi-scenario SPAROW StochasticProgram with a
   *trivial* builder (newsvendor-style) to confirm SPAROW multi-scenario
   plumbing (bundles, probabilities, maps)
5. Build a SPAROW SP whose model_builder calls EGRET for multiple
   scenarios; enforce probability hygiene and cross-bundle first-stage
   structural consistency (the foundation of non-anticipativity for
   Benders complicating-variable maps)
6. **Solve the extensive form** – reference objective (probability-
   weighted) and proof that the multi-scenario SP is a correctly formed
   MIP.  This is the critical control point.
7. (Optional) Peek at SPAROW master/subproblem transforms across bundles
8. Benders solve + automatic comparison against the multi-scenario EF
   objective (must match within tolerance)

Non-anticipativity note (important)
-----------------------------------
- Progressive Hedging in SPAROW can relax non-anticipativity.
- The extensive form (compact_repn=True) and classical Benders paths
  both *require* non-anticipativity.
  * Compact EF: only one shared copy of each first-stage variable.
  * Benders: the complicating-variable maps (int_to_FirstStageVar +
    ComponentMap) must be stable so that every subproblem is fixed to
    the *same* master proposal.  We therefore enforce a structural
    pre-solve gate that the maps have identical cardinality, domains
    and ComponentUID patterns across all bundles.  We do *not* assume
    uniqueness of the minimizer; SPAROW is designed to surface
    alternative solutions.

Each stage function returns inspectable objects and prints diagnostics.
Failures raise with stage-specific messages.

Do not attempt Stage 8 until Stage 6 has produced a finite EF objective.
"""

from __future__ import annotations

import copy
import logging
import pprint
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
# Stage 0 – Environment (identical to single-scenario version)
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
# Stage 1 – Load EGRET instance and build a model (base ModelData)
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
    """
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
    import os
    import egret

    candidates = []
    pkg_dir = os.path.dirname(egret.__file__)
    candidates.append(
        os.path.join(pkg_dir, "models", "tests", "uc_test_instances", f"{instance_name}.json")
    )
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
    Load an EGRET UC test instance and create a ConcreteModel (the *base*
    ModelData that will later be scaled / perturbed into multiple scenarios).
    """
    _print(f"\n=== STAGE 1: Build EGRET model ({instance_name}) ===")
    from egret.data.model_data import ModelData
    from egret.models.unit_commitment import create_tight_unit_commitment_model
    import os

    md = None
    source = None

    if force_synthetic:
        T = n_periods if n_periods is not None else 4
        _print(f"  force_synthetic=True → 2-gen copperplate UC, T={T}")
        md = _make_synthetic_tiny_uc(n_periods=T)
        source = f"synthetic(T={T})"
        n_periods = None

    if md is None and explicit_path is not None:
        _print(f"  Trying explicit path: {explicit_path}")
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(
                f"Stage 1 FAILED – explicit_path does not exist: {explicit_path}"
            )
        md = ModelData.read(explicit_path)
        source = explicit_path

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

    if md is None and allow_github_fetch:
        path = _fetch_egret_test_json_from_github(instance_name)
        if path is not None:
            _print(f"  Reading downloaded file {path}")
            md = ModelData.read(path)
            source = f"github:{path}"

    if md is None:
        if not allow_synthetic:
            raise FileNotFoundError(
                "Stage 1 FAILED – could not locate or download a test instance, "
                "and allow_synthetic=False."
            )
        T = n_periods if n_periods is not None else 4
        _print(f"  Falling back to synthetic ModelData (2 gens, T={T}).")
        md = _make_synthetic_tiny_uc(n_periods=T)
        source = f"synthetic(T={T})"
        n_periods = None

    if n_periods is not None:
        _print(f"  Truncating time horizon to first {n_periods} periods")
        md = _truncate_model_data_horizon(md, n_periods)
        source = f"{source}|T={n_periods}"

    print(f"  ModelData source: {source}")
    print(f"  system keys: {list(md.data.get('system', {}).keys())}")
    print(f"  element types: {list(md.data.get('elements', {}).keys())}")

    # copperplate avoids uncopyable PTDF objects that break SPAROW deepcopy
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

    keywords = (
        "uniton", "unitstart", "unitstop",
        "startup", "shutdown", "status",
        "regulation",
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
# Stage 4 – Minimal multi-scenario SPAROW SP with a trivial builder
# ---------------------------------------------------------------------------

def stage4_trivial_multi_sparow_sp():
    """
    Build a SPAROW StochasticProgram with *multiple* scenarios using the
    official newsvendor pattern.  Confirms that SPAROW multi-scenario
    plumbing (bundles, probabilities, maps) is healthy before EGRET.
    """
    _print("\n=== STAGE 4: Trivial multi-scenario SPAROW SP (newsvendor-style) ===")
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

    # Explicit multi-scenario with non-uniform probabilities
    model_data = {
        "scenarios": [
            {"ID": "low",  "d": 15, "Probability": 0.25},
            {"ID": "med",  "d": 60, "Probability": 0.50},
            {"ID": "high", "d": 72, "Probability": 0.25},
        ]
    }

    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data={})
    # Do NOT pass name=...  Leave it at the default (None).
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

    # Probability hygiene
    probs = []
    for b in sp.bundles:
        p = getattr(sp.bundles[b], "probability", None)
        probs.append(p)
        print(f"  bundle {b!r}: probability={p}")
    if any(p is None or p <= 0 for p in probs):
        raise RuntimeError("Stage 4 FAILED – missing or non-positive probability on a bundle")
    s = sum(probs)
    if abs(s - 1.0) > 1e-8:
        raise RuntimeError(f"Stage 4 FAILED – probabilities sum to {s}, expected 1.0")

    # Force creation of one bundle EF so that cuid maps are populated
    b0 = next(iter(sp.bundles))
    try:
        m = sp.create_bundle_EF(b=b0, w=None, x_bar=None, rho=None, cached=False)
        print(f"  create_bundle_EF succeeded for bundle {b0}")
    except Exception as e:
        raise RuntimeError(
            f"Stage 4 FAILED – create_bundle_EF raised: {e}"
        ) from e

    if hasattr(sp, "int_to_FirstStageVar") and b0 in sp.int_to_FirstStageVar:
        fs_map = sp.int_to_FirstStageVar[b0]
        _print(f"  int_to_FirstStageVar[{b0}]: {len(fs_map)} entries")
        for i, v in list(fs_map.items())[:3]:
            print(f"    {i}: {v.name} = {v}")
    else:
        print("  WARNING: int_to_FirstStageVar not yet populated (may be lazy)")

    _print("  Stage 4 PASSED (multi-scenario trivial SP)")
    return sp


# ---------------------------------------------------------------------------
# Stage 5 – Multi-scenario SPAROW SP whose builder calls EGRET
# ---------------------------------------------------------------------------

def _scale_md(md, scale, scen_id):
    """Deep-copy and scale all time-series loads; attach scenario_id."""
    md = copy.deepcopy(md)
    for load in md.data.get("elements", {}).get("load", {}).values():
        if "p_load" in load and isinstance(load["p_load"], dict) \
                and load["p_load"].get("data_type") == "time_series":
            load["p_load"] = dict(load["p_load"])
            load["p_load"]["values"] = [v * scale for v in load["p_load"]["values"]]
    md.data.setdefault("system", {})["scenario_id"] = scen_id
    return md


def stage5_egret_sparow_sp_multi(
    base_md,
    first_stage_names: Sequence[str],
    scenarios: Optional[List[Dict[str, Any]]] = None,
    formulation: str = "tight",
):
    """
    Construct a multi-scenario SPAROW SP that uses an EGRET model_builder.

    Parameters
    ----------
    scenarios : list of dicts
        Each dict must contain at least:
          - "ID"          : scenario identifier (used as bundle key material)
          - "Probability" : float > 0
          - "load_scale"  : float (applied to the base ModelData loads)
        Additional keys are passed through to the builder.

    Success signature
    -----------------
    - bundles non-empty and match the supplied scenarios
    - probabilities sum to 1
    - int_to_FirstStageVar non-empty and *structurally identical* across bundles
    - create_bundle_EF succeeds for every bundle
    """
    _print("\n=== STAGE 5: Multi-scenario SPAROW SP with EGRET model_builder ===")
    from sparow.sp import stochastic_program
    from egret.models.unit_commitment import create_tight_unit_commitment_model

    if scenarios is None:
        # Sensible multi-scenario default: 3 scenarios, non-uniform probs, mild load variation
        scenarios = [
            {"ID": "low",  "Probability": 0.25, "load_scale": 0.85},
            {"ID": "med",  "Probability": 0.50, "load_scale": 1.00},
            {"ID": "high", "Probability": 0.25, "load_scale": 1.15},
        ]

    # Attach base_md to every scenario so the builder can scale it
    scen_list = []
    for s in scenarios:
        entry = dict(s)
        entry["base_md"] = base_md
        scen_list.append(entry)

    model_data = {"data": {}, "scenarios": scen_list}

    def egret_builder(data, args=None):
        scale = float(data.get("load_scale", 1.0))
        base = data.get("base_md")
        sid = data.get("ID")
        if base is None:
            raise KeyError("egret_builder expected 'base_md' in scenario data")
        md = _scale_md(base, scale, sid)
        m = create_tight_unit_commitment_model(
            md, network_constraints="copperplate_power_flow", relaxed=False
        )
        # Alias objective for SPAROW robustness.
        # Prefer *deleting* the original rather than merely deactivating it;
        # a deactivated Objective can still trip SPAROW's replace-variables
        # transform under multi-scenario compact_repn=True.
        objs = list(m.component_data_objects(pyo.Objective, active=True))
        if objs and objs[0].name not in ("o", "obj"):
            original = objs[0]
            m.o = pyo.Objective(expr=original.expr, sense=original.sense)
            m.del_component(original)
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

    # ------------------------------------------------------------------
    # Probability hygiene (hard gate)
    # ------------------------------------------------------------------
    probs = []
    for b in sp.bundles:
        p = getattr(sp.bundles[b], "probability", None)
        probs.append((b, p))
        print(f"  bundle {b!r}: probability={p}")
    if any(p is None or p <= 0 for _, p in probs):
        raise RuntimeError(
            "Stage 5 FAILED – missing or non-positive probability on a bundle. "
            "Supply explicit Probability in every scenario dict."
        )
    s = sum(p for _, p in probs)
    if abs(s - 1.0) > 1e-8:
        raise RuntimeError(
            f"Stage 5 FAILED – probabilities sum to {s}, expected 1.0 "
            "(within 1e-8).  Check the Probability fields you supplied."
        )
    _print(f"  Probability sum = {s:.10f}  (OK)")

    # ------------------------------------------------------------------
    # Force map population by building every bundle EF
    # ------------------------------------------------------------------
    for b in sp.bundles:
        print(f"  Building EF for bundle {b} ...")
        try:
            m = sp.create_bundle_EF(b=b, w=None, x_bar=None, rho=None, cached=False)
        except Exception as e:
            raise RuntimeError(
                f"Stage 5 FAILED – create_bundle_EF / EGRET builder raised for "
                f"bundle {b!r}:\n  {e}"
            ) from e

    # ------------------------------------------------------------------
    # Structural first-stage map consistency across bundles
    # (foundation of non-anticipativity for Benders complicating maps)
    #
    # SPAROW prefixes every variable with the scenario/block identifier
    # (e.g. s[None,low].UnitOn[G1,1] vs s[None,med].UnitOn[G1,1]).
    # Those full names *must* differ.  What must be identical is the
    # local structure (base component + indices) and the integer keys
    # used by int_to_FirstStageVar.  We therefore compare *local* names
    # after stripping the leading "s[...]." prefix.
    # ------------------------------------------------------------------
    def _local_name(v):
        """Strip the SPAROW scenario/block prefix from a VarData name."""
        name = v.name
        # Typical form: s[None,<id>].UnitOn[G1,1]  or  s[<id>].UnitOn[...]
        if name.startswith("s["):
            idx = name.find("].")
            if idx != -1:
                return name[idx + 2 :]
        return name

    if not hasattr(sp, "int_to_FirstStageVar"):
        raise RuntimeError(
            "Stage 5 FAILED – int_to_FirstStageVar missing after create_bundle_EF."
        )

    ref_b = next(iter(sp.bundles))
    ref_map = sp.int_to_FirstStageVar.get(ref_b)
    if ref_map is None or len(ref_map) == 0:
        raise RuntimeError(
            f"Stage 5 FAILED – first-stage map for reference bundle {ref_b!r} "
            "is empty.  The names in first_stage_variables were not found as "
            "Var components in the EGRET model.  Check Stage 2 output."
        )

    ref_n = len(ref_map)
    ref_local = sorted(_local_name(v) for v in ref_map.values())
    ref_domains = sorted(( _local_name(v), str(v.domain) ) for v in ref_map.values())
    _print(f"  Reference bundle {ref_b!r}: {ref_n} first-stage variables")
    for i, v in list(ref_map.items())[:5]:
        print(f"    [{i}] {v.name}  (local={_local_name(v)})  domain={v.domain}")

    for b in sp.bundles:
        fs_map = sp.int_to_FirstStageVar.get(b)
        if fs_map is None:
            raise RuntimeError(
                f"Stage 5 FAILED – int_to_FirstStageVar missing for bundle {b!r}"
            )
        n = len(fs_map)
        if n != ref_n:
            raise RuntimeError(
                f"Stage 5 FAILED – first-stage map cardinality mismatch: "
                f"bundle {ref_b!r} has {ref_n} vars, bundle {b!r} has {n}. "
                "This breaks the complicating-variable maps required by "
                "classical Benders (non-anticipativity foundation)."
            )
        local = sorted(_local_name(v) for v in fs_map.values())
        if local != ref_local:
            raise RuntimeError(
                f"Stage 5 FAILED – first-stage *local* names differ across "
                f"bundles (after stripping scenario prefixes).\n"
                f"  {ref_b!r}: {ref_local[:6]}...\n"
                f"  {b!r}: {local[:6]}...\n"
                "The maps must be parallel for Benders fixing to be consistent."
            )
        domains = sorted(( _local_name(v), str(v.domain) ) for v in fs_map.values())
        if domains != ref_domains:
            raise RuntimeError(
                f"Stage 5 FAILED – first-stage domains differ across bundles.\n"
                f"  {ref_b!r} vs {b!r}"
            )

    _print(f"  Cross-bundle first-stage structural check PASSED "
           f"({ref_n} vars, identical local names/domains across "
           f"{len(sp.bundles)} bundles)")

    # Sanity: all should be binary (or integer)
    non_bin = [v for v in ref_map.values()
               if v.domain is not pyo.Binary and "binary" not in str(v.domain).lower()]
    if non_bin:
        print(f"  WARNING: {len(non_bin)} first-stage vars are not Binary")

    _print("  Stage 5 PASSED")
    return sp


# ---------------------------------------------------------------------------
# Stage 6 – Solve the multi-scenario Extensive Form (reference)
# ---------------------------------------------------------------------------

def stage6_solve_extensive_form(
    sp,
    solver_name: str = "gurobi",
    solver_options: Optional[dict] = None,
    time_limit: Optional[float] = 300,
):
    """
    Solve the extensive form of the multi-scenario SPAROW SP.
    This is the critical control point before any Benders work.

    For compact_repn=True the first-stage variables exist only once
    (shared); non-anticipativity is therefore structural.
    """
    _print("\n=== STAGE 6: Multi-scenario Extensive Form solve (reference) ===")
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
    _print("  Building and solving multi-scenario extensive form ...")

    results = None
    used_compact = True
    try:
        # Prefer compact_repn=True (shared first-stage vars).  If the
        # replace-variables transform fails on an EGRET component type
        # under multi-scenario, fall back to non-compact EF.  Both yield
        # a mathematically valid reference objective for Benders.
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
        msg = str(e)
        if "No transformation handler registered" in msg or "replace_variables" in msg.lower():
            _print(
                "  compact_repn=True failed inside SPAROW replace-variables "
                f"transform:\n    {e}"
            )
            _print("  Falling back to compact_repn=False (non-compact multi-scenario EF) ...")
            used_compact = False
            try:
                if hasattr(sp, "set_solver"):
                    sp.set_solver(solver_name)
                M = sp.create_EF(compact_repn=False)
                results = sp.solve(M, solver_options=solver_options)
            except Exception as e2:
                _print(f"  Non-compact EF also failed: {e2}")
                raise RuntimeError(
                    "Stage 6 FAILED – both compact and non-compact multi-scenario "
                    "EF construction failed.  See errors above."
                ) from e2
        else:
            _print(f"  EF solve raised: {e}")
            raise

    _print(f"  EF representation used: {'compact' if used_compact else 'non-compact'}")

    # ------------------------------------------------------------------
    # Defensive objective extraction (SPAROW result shapes vary)
    # ------------------------------------------------------------------
    obj_value = None
    extraction_errors = []

    def _try_to_dict(obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if isinstance(obj, dict):
            return obj
        return None

    # Path A: solutions pool
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

    # Path C: nested .solutions
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

    # Path D: SP helper
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
        raise RuntimeError(
            "Stage 6 FAILED – could not recover a numeric objective value from "
            "the extensive-form solve.  See the structure dump above."
        )

    _print(f"  Multi-scenario EF objective value: {obj_value:.6g}")

    # Sample first-stage values (compact EF → shared vars)
    try:
        b0 = next(iter(sp.bundles))
        if hasattr(sp, "int_to_FirstStageVar") and b0 in sp.int_to_FirstStageVar:
            fs_map = sp.int_to_FirstStageVar[b0]
            _print("  Sample first-stage values (post-EF, shared under compact_repn):")
            for i, v in list(fs_map.items())[:8]:
                _print(f"    {v.name} = {pyo.value(v)}")
    except Exception as e:
        _print(f"  (Could not print first-stage values: {e})")

    _print("  Stage 6 PASSED – multi-scenario reference objective recorded")
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
    Call SPAROW's internal transform helpers on one (or more) bundles and
    report residual discrete structure and master etas *before* any Benders
    solve.
    """
    _print("\n=== STAGE 7: Peek at SPAROW transforms (no solve) ===")
    from sparow.benders import BendersSolver

    b0 = next(iter(sp.bundles))
    print(f"  Using representative bundle {b0}")

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
        if n_bin > 0:
            print("  WARNING: residual Binary > 0 – classical dual cuts will fail "
                  "unless these are relaxed or moved into the first-stage list.")
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
            print(f"  Master etas (one per bundle): {list(master.etas.keys())}")
        objs = list(master.component_data_objects(pyo.Objective, active=True))
        print(f"  Master active objectives: {[o.name for o in objs]}")
    except Exception as e:
        print(f"  Master transform raised (inspect carefully): {e}")

    print("  Stage 7 finished (manual inspection of any warnings above)")
    return True


# ---------------------------------------------------------------------------
# Stage 8 – Benders, with mandatory comparison to multi-scenario EF objective
# ---------------------------------------------------------------------------

def stage8_run_benders(
    sp,
    ef_reference: dict,
    max_iterations: int = 80,
    eta_lower: float = 0.0,
    obj_tol: float = 1e-3,
    rel_tol: float = 1e-4,
):
    """
    Run SPAROW BendersSolver on the multi-scenario SP and compare the obtained
    objective against the extensive-form reference from Stage 6.

    All original work-arounds (remove_first_stage_only_cons, residual Binary
    relaxation, first-stage seeding/unfixing, allow_infeasible, persistent
    solvers) are retained because they are required for EGRET UC models.
    """
    _print("\n=== STAGE 8: SPAROW BendersSolver (multi-scenario + EF comparison) ===")
    from sparow.benders import BendersSolver

    ef_obj = ef_reference["objective"]
    _print(f"  Multi-scenario EF reference objective: {ef_obj:.6g}")

    # ------------------------------------------------------------------
    # eta bounds: every bundle must have an explicit finite lower bound.
    # ------------------------------------------------------------------
    if eta_lower is None:
        eta_lower = 0.0
    eta_bounds_map = {b: (float(eta_lower), None) for b in sp.bundles}
    _print(f"  eta_bounds_map keys: {list(eta_bounds_map.keys())}")
    for b, bounds in eta_bounds_map.items():
        if bounds[0] is None:
            raise RuntimeError(
                f"Stage 8 FAILED – eta lower bound for bundle {b!r} is None. "
                "Provide a finite default (e.g. 0.0 or -1e6)."
            )

    # ------------------------------------------------------------------
    # First-stage initialization + unfix (applied to *every* bundle)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Monkey-patch SPAROW's subproblem setup (same as single-scenario)
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
        # Classical Benders needs an LP subproblem for duals.
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

    BendersSolver._setup_topas_subproblem = staticmethod(
        _setup_with_remove_fs_only_cons
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

    # ------------------------------------------------------------------
    # Per-iteration progress banners
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
            f"multi-scenario EF reference {ef_obj:.6g} by more than tolerance "
            f"{tol:.6g}.  Because the EF solved successfully, the discrepancy "
            "is almost certainly in the Benders path (transforms, cut "
            "generation, probability weighting, first-stage cost handling, "
            "or eta bounds)."
        )

    _print("  Stage 8 PASSED – multi-scenario Benders matches EF within tolerance")
    return {
        "benders_objective": benders_obj,
        "ef_objective": ef_obj,
        "abs_diff": abs_diff,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Driver: run stages sequentially, stop at first failure
# ---------------------------------------------------------------------------

def run_staged_diagnostics_multi(
    instance_name: str = "tiny_uc_tc",
    first_stage_names: Optional[Sequence[str]] = None,
    scenarios: Optional[List[Dict[str, Any]]] = None,
    stop_after_ef: bool = True,
    run_benders: bool = False,
    ef_solver: str = "gurobi",
    ef_time_limit: float = 300,
    explicit_path: Optional[str] = None,
    allow_synthetic: bool = True,
    allow_github_fetch: bool = True,
    force_synthetic: bool = True,
    n_periods: Optional[int] = 4,
    max_benders_iterations: int = 80,
):
    """
    Execute the multi-scenario staged path.

    Recommended first run (synthetic, 3 scenarios, non-uniform probs)::

        run_staged_diagnostics_multi(
            force_synthetic=True,
            n_periods=4,
            first_stage_names=["UnitOn"],
            scenarios=[
                {"ID": "low",  "Probability": 0.25, "load_scale": 0.85},
                {"ID": "med",  "Probability": 0.50, "load_scale": 1.00},
                {"ID": "high", "Probability": 0.25, "load_scale": 1.15},
            ],
            stop_after_ef=False,
            run_benders=True,
            max_benders_iterations=80,
        )

    Parameters
    ----------
    scenarios : list of dicts (optional)
        Each entry needs ID, Probability (>0), load_scale.  If None, a
        default 3-scenario non-uniform set is used.
    force_synthetic / n_periods : control the base UC size.
    max_benders_iterations : higher default than the single-scenario script
        because multi-scenario UC typically needs more cuts.
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

    stage4_trivial_multi_sparow_sp()

    if scenarios is None:
        scenarios = [
            {"ID": "low",  "Probability": 0.25, "load_scale": 0.85},
            {"ID": "med",  "Probability": 0.50, "load_scale": 1.00},
            {"ID": "high", "Probability": 0.25, "load_scale": 1.15},
        ]
        _print(f"\n  Using default multi-scenario set: {scenarios}")

    sp = stage5_egret_sparow_sp_multi(
        md,
        first_stage_names=first_stage_names,
        scenarios=scenarios,
    )

    # ------------------------------------------------------------------
    # Critical control point: solve the multi-scenario extensive form.
    #
    # SPAROW's EF path (especially the single_bundle scheme used by the
    # non-compact fallback) can mutate the SP's bundling so that later
    # Benders transforms, which require len(bundle.scenarios) == 1, fail.
    # Full deepcopy of an EGRET-backed SP is also unreliable (uncopyable
    # Contingencies / _init_values fields, pickle dict_keys errors).
    #
    # Therefore: solve EF on the current SP, record the reference
    # objective, then *rebuild* a fresh multi-scenario SP for Stages 7–8.
    # Stage 5 is cheap on the synthetic instance, so this is robust.
    # ------------------------------------------------------------------
    ef_ref = stage6_solve_extensive_form(
        sp,
        solver_name=ef_solver,
        time_limit=ef_time_limit,
    )

    if stop_after_ef and not run_benders:
        print("\n*** Multi-scenario staged diagnostics complete through EF (Stage 6). ***")
        print(f"*** EF reference objective = {ef_ref['objective']:.6g} ***")
        print("*** Re-run with run_benders=True only after this stage is clean. ***")
        return {"sp": sp, "ef_reference": ef_ref}

    # Rebuild a clean multi-bundle SP for Benders (EF may have mutated bundling)
    _print("\n  Rebuilding multi-scenario SP for Benders (EF may have altered bundles) ...")
    sp = stage5_egret_sparow_sp_multi(
        md,
        first_stage_names=first_stage_names,
        scenarios=scenarios,
    )

    # Optional transform inspection
    stage7_peek_transforms(sp)

    if not run_benders:
        print("\n*** Multi-scenario EF + transform peek done. "
              "Set run_benders=True for Stage 8. ***")
        return {"sp": sp, "ef_reference": ef_ref}

    # Benders on the freshly rebuilt multi-bundle SP
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
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Multi-scenario staged validation of an EGRET unit-commitment model "
            "under SPAROW classical Benders.  Defaults to the recommended "
            "synthetic 3-scenario non-uniform path."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model size / source
    parser.add_argument(
        "--force-synthetic", action="store_true", default=True,
        help="Use the minimal 2-gen copperplate synthetic ModelData",
    )
    parser.add_argument(
        "--no-force-synthetic", action="store_false", dest="force_synthetic",
        help="Allow official / GitHub / explicit instances",
    )
    parser.add_argument(
        "--n-periods", type=int, default=4,
        help="Horizon length (synthetic size or truncation of a real instance)",
    )
    parser.add_argument(
        "--instance-name", type=str, default="tiny_uc_tc",
        help="EGRET test instance name (ignored when --force-synthetic)",
    )
    parser.add_argument(
        "--explicit-path", type=str, default=None,
        help="Path to a local EGRET JSON instance",
    )

    # First-stage selection
    parser.add_argument(
        "--first-stage-names", type=str, default=None,
        help="Comma-separated first-stage component names "
             "(default: heuristic / UnitOn for synthetic)",
    )

    # Multi-scenario definition (simple scales + probs)
    parser.add_argument(
        "--scales", type=str, default="0.85,1.00,1.15",
        help="Comma-separated load scales (one per scenario)",
    )
    parser.add_argument(
        "--probs", type=str, default="0.25,0.50,0.25",
        help="Comma-separated probabilities (must sum to 1)",
    )
    parser.add_argument(
        "--scenario-ids", type=str, default="low,med,high",
        help="Comma-separated scenario IDs",
    )

    # Control flow
    parser.add_argument(
        "--stop-after-ef", action="store_true", default=False,
        help="Stop after Stage 6 (EF) even if --run-benders is set",
    )
    parser.add_argument(
        "--run-benders", action="store_true", default=True,
        help="Continue to Stage 8 (Benders) after a successful EF",
    )
    parser.add_argument(
        "--no-run-benders", action="store_false", dest="run_benders",
        help="Stop after EF / transform peek (do not call Benders)",
    )

    # Solvers / limits
    parser.add_argument(
        "--ef-solver", type=str, default="gurobi",
        help="Solver for the extensive-form reference",
    )
    parser.add_argument(
        "--ef-time-limit", type=float, default=300.0,
        help="Time limit (seconds) for the EF solve",
    )
    parser.add_argument(
        "--max-benders-iterations", type=int, default=80,
        help="Maximum Benders iterations (higher than single-scenario default)",
    )

    args = parser.parse_args()

    # Build first_stage_names list
    fs_names = None
    if args.first_stage_names:
        fs_names = [s.strip() for s in args.first_stage_names.split(",") if s.strip()]

    # Build scenarios list from scales / probs / ids
    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    probs  = [float(x) for x in args.probs.split(",")  if x.strip()]
    ids    = [s.strip() for s in args.scenario_ids.split(",") if s.strip()]
    if not (len(scales) == len(probs) == len(ids)):
        parser.error(
            f"--scales, --probs and --scenario-ids must have the same length "
            f"(got {len(scales)}, {len(probs)}, {len(ids)})"
        )
    if abs(sum(probs) - 1.0) > 1e-8:
        parser.error(f"--probs must sum to 1.0 (got {sum(probs)})")
    scenarios = [
        {"ID": sid, "Probability": p, "load_scale": sc}
        for sid, p, sc in zip(ids, probs, scales)
    ]

    print("=== Multi-scenario staged runner (CLI) ===")
    print(f"  force_synthetic        = {args.force_synthetic}")
    print(f"  n_periods              = {args.n_periods}")
    print(f"  first_stage_names      = {fs_names}")
    print(f"  scenarios              = {scenarios}")
    print(f"  stop_after_ef          = {args.stop_after_ef}")
    print(f"  run_benders            = {args.run_benders}")
    print(f"  ef_solver              = {args.ef_solver}")
    print(f"  max_benders_iterations = {args.max_benders_iterations}")
    print()

    run_staged_diagnostics_multi(
        instance_name=args.instance_name,
        first_stage_names=fs_names,
        scenarios=scenarios,
        stop_after_ef=args.stop_after_ef,
        run_benders=args.run_benders,
        ef_solver=args.ef_solver,
        ef_time_limit=args.ef_time_limit,
        explicit_path=args.explicit_path,
        force_synthetic=args.force_synthetic,
        n_periods=args.n_periods,
        max_benders_iterations=args.max_benders_iterations,
    )
