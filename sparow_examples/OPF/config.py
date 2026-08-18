from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    seed: int = 55
    dr: float = 1e7
    max_iterations: int = 20
    scenarios: tuple = ("1", "2", "3", "4", "5", "6", "7", "8")
    bound: float = 0.1
    data_dir: str = '../../../../pglib-opf/pglib_opf_case118_ieee.m'
