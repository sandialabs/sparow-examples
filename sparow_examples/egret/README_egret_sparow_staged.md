# EGRET + SPAROW Classical Benders — Staged Diagnostics

This guide documents `egret_sparow_staged.py`: a staged diagnostic driver that builds an EGRET unit-commitment (UC) model, wraps it as a SPAROW stochastic program, solves the extensive form (EF) for a reference objective, then runs classical Benders via SPAROW / OR-Topas and compares the result to the EF.

The script is intentionally **staged** so each layer can fail with a clear signature before the full Benders path is exercised.

---

## Prerequisites

| Package | Role |
|---------|------|
| **EGRET** (`gridx-egret`) | UC formulation (`create_tight_unit_commitment_model`) |
| **Pyomo** | Modeling layer |
| **SPAROW** | Two-stage SP object, EF solver, BendersSolver |
| **OR-Topas** | Subproblem transform (`standard_lp`) and cut generation |
| **Gurobi** | MIP / LP solver (`gurobi` / `gurobi_persistent`) |

### EGRET install (important)

Conda / PyPI installs of EGRET often **omit** `egret/models/tests/uc_test_instances/`. The official `tiny_uc_tc.json` lives only in the full source tree.

**Recommended:** install EGRET editable from GitHub (or a local clone):

```bash
# Optional: isolate from packages that pin an older EGRET (e.g. gridx-prescient)
conda create -n egret_experiments --clone <your_existing_env>
conda activate egret_experiments

# Prefer a clean EGRET from source
pip uninstall gridx-egret -y   # if a conda/pip build is present
git clone https://github.com/grid-parity-exchange/Egret.git
cd Egret
pip install -e .
```

Confirm the test instance is visible:

```bash
python -c "import egret, os; print(os.path.join(os.path.dirname(egret.__init__.__file__), 'models', 'tests', 'uc_test_instances', 'tiny_uc_tc.json'))"
```

If that path does not exist, either use a source install or pass `force_synthetic=True` (in-code 2-gen model; no JSON required).

### SPAROW / OR-Topas

Install from your lab’s SPAROW and OR-Topas source trees (editable installs are typical for development). Both must import cleanly:

```bash
python -c "import sparow; import or_topas; print('OK')"
```

### Gurobi

A working Gurobi license is required. The driver uses `gurobi` for the EF and `gurobi_persistent` for Benders master and subproblems.

---

## What the stages check

| Stage | Purpose | Success signature |
|-------|---------|-------------------|
| **0** | Imports (EGRET, SPAROW, OR-Topas) | All packages found |
| **1** | Build EGRET UC model | ConcreteModel + active objective |
| **2** | List Binary / continuous vars | Print suggested first-stage names |
| **3** | Verify `first_stage_names` | All names present; count VarData objects |
| **4** | Trivial SPAROW SP (newsvendor-style) | `initialize_model` with `name=None` (default) |
| **5** | EGRET model_builder → SPAROW SP | `int_to_FirstStageVar` populated |
| **6** | Extensive form solve | Numeric EF objective (reference) |
| **7** | Peek master / subproblem transforms | Residual Binary count ≈ 0 |
| **8** | BendersSolver + EF comparison | \|Benders − EF\| within tolerance |

Stages 0–6 should pass before interpreting Stage 8 failures as Benders bugs.

---

## Critical configuration rules (learned the hard way)

1. **All Binary variables must be first-stage** for classical Benders with this stack.  
   Residual binaries in the subproblem make it a MIP → no duals (`Pi`) and no Farkas rays.  
   For `tiny_uc_tc` include at least:
   ```python
   first_stage_names=['UnitOn', 'UnitStart', 'UnitStop', 'StartupIndicator', 'RegulationOn']
   ```

2. **Network model:** default to `copperplate_power_flow`.  
   Default `ptdf_power_flow` attaches non-cloneable `_PTDFs` / `VirtualPTDFMatrix` objects and floods clone warnings. Reintroduce PTDF only after the copperplate path is solid.

3. **`remove_first_stage_only_cons=True`** on the subproblem transform.  
   Pure first-stage constraints (commitment logic, initial status) must stay on the master only; otherwise the OR-Topas `standard_lp` transform can build trivial `True` constraints.

4. **`allow_infeasible_subproblems=True`**.  
   Early master proposals are often infeasible; feasibility cuts require Farkas duals.

5. **Persistent solvers** for both master and subproblem when `is_persistent_solver=True`:
   ```python
   solver="gurobi_persistent"
   subproblem_solver="gurobi_persistent"
   ```

6. **Eta bounds:** every bundle needs a finite lower bound (e.g. `0.0` for non-negative costs).

7. **SPAROW model name:** leave the default `None` (do not pass names like `"news"` / `"uc"` that break Benders expectations).

---

## Command-line recipes

All commands assume you are in the directory that contains `egret_sparow_staged.py` (or that it is on `PYTHONPATH`), with the correct conda env activated.

### Option A — synthetic 2-gen × 4-period (fastest correctness check)

No EGRET test JSON required. Expected: converge in a handful of iterations; Benders = EF.

```bash
python -c "
from egret_sparow_staged import run_staged_diagnostics
run_staged_diagnostics(
    force_synthetic=True,
    n_periods=4,
    first_stage_names=['UnitOn', 'UnitStart', 'UnitStop'],
    scenario_scales={'base': 1.0},
    stop_after_ef=False,
    run_benders=True,
    max_benders_iterations=30,
    ef_solver='gurobi',
    ef_time_limit=60,
)
" 2>&1 | tee egret_benders_optionA.log
```

**Validated result:** Benders = EF = 5886.4 (9 iterations, 8 cuts).

---

### Option B — real `tiny_uc_tc`, 4 periods

Requires EGRET source install so `tiny_uc_tc.json` is present.

```bash
python -c "
from egret_sparow_staged import run_staged_diagnostics
run_staged_diagnostics(
    instance_name='tiny_uc_tc',
    n_periods=4,
    first_stage_names=['UnitOn', 'UnitStart', 'UnitStop', 'StartupIndicator', 'RegulationOn'],
    scenario_scales={'base': 1.0},
    stop_after_ef=False,
    run_benders=True,
    max_benders_iterations=300,
    ef_solver='gurobi',
    ef_time_limit=120,
)
" 2>&1 | tee egret_benders_optionB.log
```

**Validated result:** Benders = EF = 69376 (within 300 iterations; residual Binary = 0).

---

### Option B2 — real `tiny_uc_tc`, 8 periods (2× Option B)

```bash
python -c "
from egret_sparow_staged import run_staged_diagnostics
run_staged_diagnostics(
    instance_name='tiny_uc_tc',
    n_periods=8,
    first_stage_names=['UnitOn', 'UnitStart', 'UnitStop', 'StartupIndicator', 'RegulationOn'],
    scenario_scales={'base': 1.0},
    stop_after_ef=False,
    run_benders=True,
    max_benders_iterations=600,
    ef_solver='gurobi',
    ef_time_limit=180,
)
" 2>&1 | tee egret_benders_optionB2_T8.log
```

---

### Option C — full 24-period `tiny_uc_tc`

```bash
python -c "
from egret_sparow_staged import run_staged_diagnostics
run_staged_diagnostics(
    instance_name='tiny_uc_tc',
    first_stage_names=['UnitOn', 'UnitStart', 'UnitStop', 'StartupIndicator', 'RegulationOn'],
    scenario_scales={'base': 1.0},
    stop_after_ef=False,
    run_benders=True,
    max_benders_iterations=1000,
    ef_solver='gurobi',
    ef_time_limit=300,
)
" 2>&1 | tee egret_benders_optionC.log
```

Expect a much larger first-stage binary set (~1000+) and a long iteration budget. Prefer confirming Options A → B → B2 first.

---

### EF-only dry run (no Benders)

Useful when debugging model build / first-stage lists:

```bash
python -c "
from egret_sparow_staged import run_staged_diagnostics
run_staged_diagnostics(
    instance_name='tiny_uc_tc',
    n_periods=4,
    first_stage_names=['UnitOn', 'UnitStart', 'UnitStop', 'StartupIndicator', 'RegulationOn'],
    stop_after_ef=True,
    run_benders=False,
    ef_solver='gurobi',
    ef_time_limit=120,
)
" 2>&1 | tee egret_ef_only.log
```

---

## How to read the log

**Healthy Stage 7**

```text
  Subproblem residual Binary vars (should be 0 or few): 0
```

**Healthy Stage 8**

```text
  --- Benders iteration k  cuts_added=...  cuts_total=...  ---
  ...
  Benders objective value: <number>
  |Benders - EF| = 0  (tol = ...)
  Stage 8 PASSED – Benders matches EF within tolerance
```

**Common failure modes**

| Symptom | Likely cause |
|---------|----------------|
| `FileNotFoundError` for `tiny_uc_tc.json` | EGRET not installed from source; use source install or `force_synthetic=True` |
| Residual Binary ≫ 0 | Missing names in `first_stage_names` |
| `Unable to retrieve attribute 'Pi'` | Subproblem still MIP (residual binaries) |
| `Unable to retrieve attribute 'FarkasDual'` | Infeasible MIP subproblem; need LP + `allow_infeasible` |
| `trivial Boolean (True)` in `aux_cons` | Pure first-stage constraints in subproblem; need `remove_first_stage_only_cons=True` |
| `GUROBIFILE` has no `set_instance` | Non-persistent solver with `is_persistent_solver=True` |
| Benders ≪ EF after many iterations | Incomplete lower bound (raise `max_benders_iterations`) or missing first-stage costs |
| PTDF / `_PTDFs` clone flood | Using `ptdf_power_flow`; switch to copperplate for validation |

---

## API sketch

```python
from egret_sparow_staged import run_staged_diagnostics

run_staged_diagnostics(
    instance_name="tiny_uc_tc",   # or ignored if force_synthetic=True
    force_synthetic=False,
    n_periods=None,              # truncate horizon; None = full
    first_stage_names=[...],     # required for classical Benders
    scenario_scales={"base": 1.0},
    stop_after_ef=False,
    run_benders=True,
    max_benders_iterations=300,
    ef_solver="gurobi",
    ef_time_limit=120,
    explicit_path=None,          # optional path to a UC JSON
    allow_synthetic=True,
    allow_github_fetch=True,
)
```

---

## Design notes for future work

- **Stochastic extension:** SPAROW’s SP + scenario `model_builder` is the intended path for multi-scenario UC; the current recipes use a single scenario (`base`) to validate the Benders transform against the EF.
- **PTDF / network:** After copperplate is reliable, reintroduce `ptdf_power_flow` only with a strategy that avoids non-cloneable attributes (or strip them before deepcopy).
- **Termination:** SPAROW may hit `max_iterations` while the master incumbent already matches the EF (cuts still generated). Comparing the recovered objective to the EF remains the correctness check.
- **Warm start:** Seeding first-stage variables from the EF solution can reduce early feasibility-cut iterations (optional enhancement).

---

## File layout (expected)

```text
egret_sparow_staged.py          # staged driver
README_egret_sparow_staged.md   # this guide
egret_benders_optionA.log       # example logs (optional)
egret_benders_optionB.log
...
```

Place this README next to the driver in your experiments repo so teammates can reproduce the same validation ladder (A → B → B2 → C) without rediscovering the configuration constraints above.
