from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    seed: int = 55
    # dr: float = 1e7
    # max_iterations: int = 20
    scenarios: tuple = tuple(str(i) for i in range(100))
    bound: float = 0.25
    data_dir: str = '../../../pglib-opf/pglib_opf_case30_ieee.m'

    # Preferred LF model for UQ / ACV-MRP experiments.
    # Supported values below: "dcopf", "copperplate"
    lf_model_type: str = "dcopf"
