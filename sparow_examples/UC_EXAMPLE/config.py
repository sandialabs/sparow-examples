from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    seed: int = 55
    dr: float = 1e7
    max_iterations: int = 20

