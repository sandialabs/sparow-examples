"""
Utilities for using the OPF multifidelity models with the SPAROW Confidence Interval code.
"""

from __future__ import annotations

from typing import List, Optional

from egret.parsers.matpower_parser import create_ModelData

from sparow_examples.uq_opf.model_builders import (
    LF_builder_multitime_period,  # copperplate
    HF_builder_multitime_period,  # DCOPF
    uHF_builder_multitime_period,  # ACOPF
)
from sparow_examples.uq_opf.config import ExperimentConfig
from sparow_examples.uq_opf.scenarios import build_model_data_time

from sparow.conf_intervals.scenario_population import FiniteScenarioPopulation
from sparow.conf_intervals.scenario_sampler import ScenarioSampler
from sparow.conf_intervals.sp_model_wrapper_for_uq import SPModelWrapperforUQ
from sparow.conf_intervals.model_ensemble import ModelEnsemble
from sparow.conf_intervals.protocols import (
    StochasticProgramModelProtocol,
    ModelEnsembleProtocol,
    ScenarioPopulationProtocol,
    ScenarioSamplerProtocol,
)


def _get_first_stage_variable_order(data_dir: str) -> List[str]:
    """
    Build the explicit ordered names of the first-stage variables.
    We use this ordering to define the first-stage candidate solution xhat
    consistently across HF and LF models.

    Parameters
    ----------
    data_dir : str
        Path to the MATPOWER / PGLib-OPF case file.

    Returns
    -------
    list of str
        Ordered first-stage variable names matching the SPAROW model output.
    """

    # Parse the base OPF case file and return generator names in a stable order.
    md = create_ModelData(data_dir)
    gen_dict = md.data["elements"]["generator"]
    gen_names = list(
        gen_dict.keys()
    )  # has ordered generator names as they appear in the parsed Egret ModelData.

    # Format returned by SPAROW
    return [f"time_periods[1].m.pg['{g}']" for g in gen_names]


def _build_shared_scenario_population(
    cfg: ExperimentConfig,
) -> FiniteScenarioPopulation:
    """
    Build the finite scenario population shared by HF and LF OPF models.

    Parameters
    ----------
    cfg : ExperimentConfig
        Experiment configuration.

    Returns
    -------
    FiniteScenarioPopulation
        Population object compatible with the confidence-interval code.
    """
    model_data_time = build_model_data_time(
        scenarios=cfg.scenarios,
        bound=cfg.bound,
        data_dir=cfg.data_dir,
        seed=cfg.seed,
    )

    scenario_population = FiniteScenarioPopulation(
        scenarios=model_data_time["scenarios"],
        required_scenario_keys=[
            "ID",
            "Probability",
            "DEMAND_1",
            "DEMAND_2",
            "DEMAND_3",
            "DEMAND_4",
            "data_dir",
        ],
        scenario_vector_keys=[
            "DEMAND_1",
            "DEMAND_2",
            "DEMAND_3",
            "DEMAND_4",
        ],
        # data_dir is fixed application metadata: it is not part of the uncertain
        # vector, but it must be restored whenever PyApprox decodes a scenario.
        fixed_metadata={
            "data_dir": cfg.data_dir,
        },
    )

    return scenario_population


def _build_shared_sampler(
    scenario_population: ScenarioPopulationProtocol,
    seed: int,
    with_replacement: bool,
) -> ScenarioSamplerProtocol:
    """
    Build one shared sampler from the supplied scenario population.

    Parameters
    ----------
    scenario_population : ScenarioPopulationProtocol
        Shared finite scenario population.
    seed : int
        Sampling seed.
    with_replacement : bool
        Whether batches are drawn with replacement.

    Returns
    -------
    ScenarioSamplerProtocol
        Shared sampler object.
    """
    return ScenarioSampler(
        scenario_population=scenario_population,
        seed=seed,
        with_replacement=with_replacement,
    )


def _select_low_fidelity_builder(lf_model_type: str):
    """
    Select the low-fidelity OPF model builder.

    Parameters
    ----------
    lf_model_type : str
        Lower-fidelity model identifier.

    Returns
    -------
    callable
        SPAROW-compatible model-builder function.

    Raises
    ------
    ValueError
        If `lf_model_type` is unsupported.
    """
    if lf_model_type == "dcopf":
        return HF_builder_multitime_period
    elif lf_model_type == "copperplate":
        return LF_builder_multitime_period
    else:
        raise ValueError(
            f"Unsupported lf_model_type={lf_model_type!r}. "
            "Expected one of {'dcopf', 'copperplate'}."
        )


def get_sp_model_for_uq(
    model_name: str = "HF",
    use_integer: bool = False,  # dummy compatibility argument; TODO: replace with more flexible kwargs handling
    seed: int = 12345,
    with_replacement: bool = True,
    lf_model_type: str = "dcopf",
    scenario_population: Optional[ScenarioPopulationProtocol] = None,
    scenario_sampler: Optional[ScenarioSamplerProtocol] = None,
) -> StochasticProgramModelProtocol:
    """
    Build one OPF stochastic-program wrapper compatible with the UQ framework.

    Parameters
    ----------
    model_name : str, optional
        Fidelity/model label. Supported values:
          - "HF" : high-fidelity ACOPF
          - "LF" : lower-fidelity DCOPF or copperplate
    use_integer: bool, optional
    seed : int, optional
        Seed for the scenario sampler.
    with_replacement : bool, optional
        Whether scenario batches are sampled with replacement.
    lf_model_type : str, optional
        Lower-fidelity model type to use when `model_name="LF"`.
    scenario_population : ScenarioPopulationProtocol, optional
        Shared scenario population object. If None, one is built internally.
    scenario_sampler : ScenarioSamplerProtocol, optional
        Shared scenario sampler object. If None, one is built internally.

    Returns
    -------
    StochasticProgramModelProtocol
        One wrapped OPF model that is compatible with the SPAROW UQ framework.
    """
    cfg = ExperimentConfig()

    if scenario_population is None:
        scenario_population = _build_shared_scenario_population(cfg)

    # The same population is used by both fidelities; only the physics differs.
    if scenario_sampler is None:
        scenario_sampler = _build_shared_sampler(
            scenario_population=scenario_population,
            seed=seed,
            with_replacement=with_replacement,
        )

    if model_name == "HF":
        model_builder = uHF_builder_multitime_period
        fidelity = "high"
    elif model_name == "LF":
        model_builder = _select_low_fidelity_builder(lf_model_type)
        fidelity = "low"
    else:
        raise ValueError(f"Unknown model_name={model_name!r}. Expected 'HF' or 'LF'.")

    first_stage_vars = ["time_periods[1].m.pg[*]"]
    first_stage_order = _get_first_stage_variable_order(cfg.data_dir)

    model = SPModelWrapperforUQ(
        name=model_name,
        fidelity=fidelity,
        scenario_population=scenario_population,
        scenario_sampler=scenario_sampler,
        model_builder=model_builder,
        # No extra app_data is needed for this OPF setup.
        app_data={},
        first_stage_variables=first_stage_vars,
        first_stage_variable_order=first_stage_order,
    )

    if not isinstance(model, StochasticProgramModelProtocol):
        raise RuntimeError(
            f"Object returned by get_sp_model_for_uq(...) for model_name={model_name} "
            "does not satisfy StochasticProgramModelProtocol."
        )

    return model


def get_model_ensemble_for_uq(
    model_name: str = "HF",
    use_integer: bool = False,  # dummy compatibility argument; TODO: replace with more flexible kwargs handling
    seed: int = 12345,
    with_replacement: bool = True,
    lf_model_type: str = "dcopf",
) -> ModelEnsembleProtocol:
    """
    Build the two-model OPF ensemble used by ACV-MRP.
    """
    cfg = ExperimentConfig()

    # Construct scenario population and sampler once and reuse in both wrappers.
    shared_population = _build_shared_scenario_population(cfg)
    shared_sampler = _build_shared_sampler(
        scenario_population=shared_population,
        seed=seed,
        with_replacement=with_replacement,
    )

    hf_model = get_sp_model_for_uq(
        model_name="HF",
        use_integer=use_integer,  # dummy compatibility argument; TODO: replace with more flexible kwargs handling
        seed=seed,
        with_replacement=with_replacement,
        lf_model_type=lf_model_type,
        scenario_population=shared_population,
        scenario_sampler=shared_sampler,
    )

    lf_model = get_sp_model_for_uq(
        model_name="LF",
        use_integer=use_integer,  # dummy compatibility argument; TODO: replace with more flexible kwargs handling
        seed=seed,
        with_replacement=with_replacement,
        lf_model_type=lf_model_type,
        scenario_population=shared_population,
        scenario_sampler=shared_sampler,
    )

    ensemble = ModelEnsemble([hf_model, lf_model])

    if not isinstance(ensemble, ModelEnsembleProtocol):
        raise RuntimeError(
            "Object returned by get_model_ensemble_for_uq(...) does not satisfy "
            "ModelEnsembleProtocol."
        )

    return ensemble
