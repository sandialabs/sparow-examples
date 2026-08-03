# This is a light wrapper that allows us to use the bootsp
# package from mpi-sppy for confidence interval construction.
# The instance being tested is Advanced Farmers
 
import pyomo.environ as pyo
from sparow.sp import stochastic_program
from sparow.ef import ExtensiveFormSolver
 
import mpisppy.utils.sputils as sputils
import mpisppy.scenario_tree as scenario_tree
 
# Import everything we need from existing module
from sparow_examples.farmers.MRPfarmers import (
    app_data,
    model_builder,
    Advanced_scendata,
    GlobalData,
)
 
# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
 
def _all_advanced_scenarios():
    """Return the full population of Advanced Farmers scenarios."""
    # This is a list. It contains one dictionary per possible population scenario.
    # Each scenario's dictionary must contain "ID", set of yields, and "Probability".
    return Advanced_scendata["scenarios"]
 
 
def _slice_scenarios(num_scens, start=None):
    """
    Return a contiguous slice of the Advanced Farmers scenario population.
 
    Parameters
    ----------
    num_scens : int
        Number of scenarios to expose.
    start : int or None
        Starting index in the population. Defaults to 0.
    """
    if start is None:
        start = 0
    scenario_list = _all_advanced_scenarios()
    stop = start + num_scens
    if stop > len(scenario_list):
        raise ValueError(f"Requested scenarios [{start}:{stop}] but only {len(scenario_list)} total scenarios are available.")
    return scenario_list[start:stop]
 
 
def _scenario_from_name(scenario_name):
    """
    Map a scenario name to its corresponding scenario dict.
    """
    scenario_list = _all_advanced_scenarios()
    try:
        idx = sputils.extract_num(scenario_name) % len(scenario_list)
        # print(f"Sampled scen: {idx}")
    except Exception as exc:
        raise ValueError(f"Could not parse scenario index from name {scenario_name}") from exc
 
    if idx < 0 or idx >= len(scenario_list):
        raise ValueError(f"Scenario index {idx} out of range for Advanced Farmers")
 
    return scenario_list[idx]
 
 
# ---------------------------------------------------------------------
# mpi-sppy / boot-sp required interface
# ---------------------------------------------------------------------
 
def scenario_creator(scenario_name, **kwargs):
    """
    Create one mpi-sppy scenario model for the Advanced Farmers problem.
 
    Parameters
    ----------
    scenario_name : str
        Scenario name.
 
    Returns
    -------
    Pyomo ConcreteModel
        One scenario model with mpi-sppy node metadata attached.
    """
    scen_dict = _scenario_from_name(scenario_name)
 
    args = {}
    if "use_integer" in kwargs:
        args["use_integer"] = kwargs["use_integer"]
 
    # Build the concrete Pyomo model for this scenario
    model = model_builder(scen_dict, args)
 
    # Construct the root node describing the first-stage structure for mpi-sppy.
    # In this 2-stage problem, the nonanticipative variables are the first-stage
    # acreage decisions DevotedAcreage[*].
    root = scenario_tree.ScenarioNode(
        name="ROOT",
        cond_prob=1.0,
        stage=1,
        cost_expression=model.FirstStageCost, # + model.SecondStageCost,
        nonant_list=[model.DevotedAcreage],
        scen_model=model,
    )
 
    # Explicitly provide the flat list of first-stage variable data objects
    # expected by mpi-sppy internals.
    root.nonant_vardata_list = [model.DevotedAcreage[c] for c in model.CROPS]
 
    # Attach the node list to the scenario model
    model._mpisppy_node_list = [root]
 
    # Store the scenario probability separately
    model._mpisppy_probability = "uniform"
 
    return model
 
 
def scenario_names_creator(num_scens, start=None):
    """
    Return the list of scenario names to use.
 
    Parameters
    ----------
    num_scens : int
        Number of scenarios requested.
    start : int or None
        Starting index into the full population of scenarios.
 
    Returns
    -------
    list of str
    """
    scenario_sublist = _slice_scenarios(num_scens, start=start)
    return [scen["ID"] for scen in scenario_sublist]
 
 
def kw_creator(cfg):
    """Return keyword arguments for scenario_creator."""
    kwargs = {}
 
    if hasattr(cfg, "use_integer"):
        kwargs["use_integer"] = cfg.use_integer
 
    return kwargs
 
 
def inparser_adder(cfg):
    """
    Add model-specific command-line options to the config object.
    """
    cfg.add_to_config(
        "use_integer",
        description="Use integer first-stage acreage variables",
        domain=bool,
        default=False,
    )
 
 
def xhat_generator(scenario_names, solver_name=None, **kwargs):
    """
    Compute a candidate first-stage solution xhat from the supplied scenarios.
 
    Parameters
    ----------
    scenario_names : list of str
        Names of scenarios to include in the EF.
    solver_name : str or None
        Solver name.
    kwargs : dict
        May contain use_integer.
 
    Returns
    -------
    dict
        xhat in the format expected by boot-sp, with a ROOT entry.
    """
    print(f"xhat_generator using solver_name={solver_name}")
    use_integer = kwargs.get("use_integer", False)
    args = {"use_integer": use_integer}
 
    # Build the scenario subset from the requested names
    scenario_sublist = [_scenario_from_name(name) for name in scenario_names]
 
    # Create a stochastic program instance over this subset
    local_model_data = {"scenarios": scenario_sublist}
 
    sp = stochastic_program(first_stage_variables=["DevotedAcreage[*]"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="Advanced",
        model_data=local_model_data,
        model_builder=model_builder,
    )
 
    solver = ExtensiveFormSolver()
    solver.set_options(solver=solver_name if solver_name is not None else "gurobi_direct")
    results = solver.solve(sp).to_dict()
 
    variables = results["solutions"][0]["variables"]
 
    xhat_root = {}
    for var in variables:
        name = var["name"]
        if name.startswith("DevotedAcreage["):
            xhat_root[name] = var["value"]
 
    return {"ROOT": xhat_root}
 
def data_sampler(record_num, cfg):
    """
    Return one data record for the smoothed bootstrap code.
 
    When cfg.use_fitted is False, this returns the historical yield data from
    the finite Advanced Farmers scenario population.
 
    When cfg.use_fitted is True, this samples each crop yield from the fitted
    distribution stored in cfg.fitted_distribution.
    """
    scenario_list = _all_advanced_scenarios()
 
    if not getattr(cfg, "use_fitted", False):
        idx = record_num % len(scenario_list)
        return scenario_list[idx]["Yield"]
 
    fitted = cfg.fitted_distribution
    return {
        "WHEAT": fitted["WHEAT"].sample_one()[0],
        "CORN": fitted["CORN"].sample_one()[0],
        "SUGAR_BEETS": fitted["SUGAR_BEETS"].sample_one()[0],
    }

def scenario_denouement(rank, scenario_name, scenario):
    """
    Optional mpi-sppy callback after scenario solve.
    This wrapper does not need any denouement logic, so this is a no-op.
    """
    return