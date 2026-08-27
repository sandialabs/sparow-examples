# `uq_opf`

These examples require:
- Egret: https://github.com/grid-parity-exchange/egret
- pglib-opf: https://github.com/power-grid-lib/pglib-opf

## Files

- `config.py`  
  Stores shared experiment settings for the OPF examples, including the scenario seed, the finite scenario set, the demand-perturbation bound, the PGLib-OPF case file path, and the preferred low-fidelity model for multifidelity workflows.

- `model_builders.py`  
  Defines the OPF model formulations used in the multifidelity workflows.  
  Includes:
  - a low-fidelity copperplate approximation,
  - a low-fidelity DCOPF model,
  - and a high-fidelity ACOPF model.

- `scenarios.py`  
  Builds the finite population of OPF scenarios that we can sample from for multifidelity workflows. Each scenario keeps first-period demand fixed at `1.0` and perturbs the demand multipliers in periods 2, 3, and 4 uniformly within a user-specified bound from config.py.  
  It also stores the case-file path needed to rebuild the Egret model data for each scenario.

- `uq_opf.py`  
  Contains the OPF models that are compatible with the confidence-interval code, including the `get_sp_model_for_uq(...)` and `get_model_ensemble_for_uq(...)` factory functions.  
  It supports multifidelity workflows.

- `demo_uq_opf.ipynb`  
  Notebook demo for running the confidence-interval workflows on the OPF examples.  
  Shows how to use the OPF model wrappers with the UQ code.

- `demo_pyapprox_opf.ipynb`  
  Notebook demo for using the PyApprox integration on the OPF examples.  
  Shows how to estimate model costs, estimate the relevant correlation, and allocate samples under a fixed computational budget.
