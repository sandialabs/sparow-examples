from dataclasses import dataclass
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent

@dataclass
class ExperimentConfig:
    seed: int = 55
    # dr: float = 1e7
    # max_iterations: int = 20
    scenarios: tuple = tuple(str(i) for i in range(100))
    bound: float = 0.25

    # PGLib-OPF case file used by the OPF example. Stored as an absolute string path.
    data_dir: str = str((_THIS_DIR / "../../../pglib-opf/pglib_opf_case30_ieee.m").resolve())

    # Preferred LF model for UQ / ACV-MRP experiments.
    # Supported values below: "dcopf", "copperplate"
    lf_model_type: str = "dcopf"
