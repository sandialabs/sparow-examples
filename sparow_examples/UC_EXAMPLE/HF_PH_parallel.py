from sparow.sp import stochastic_program
from sparow.ef import ExtensiveFormSolver
from scenarios import build_noisy_hf_scenarios
from model_builders import HF_builder, LF_builder_SHED

from sparow.ph.ph_mpisppy import (
    ProgressiveHedgingSolver_MPISPPY,
)

import pprint

dir_start='data/rts_noisey/scenarios_2/'

model_data_noisey = {
    "HF": {
        "scenarios": [
        {
            "ID": "Scen1_HF",
            "Probability": 1/8,
            "DEMAND":100,
            "data_dir":dir_start+'scenario_1.json',
        },
        {
            "ID": "Scen2_HF",
            "Probability": 1/8,
            "DEMAND":200,
            "data_dir":dir_start+'scenario_2.json',
        },
        {
            "ID": "Scen3_HF",
            "Probability": 1/8,
            "DEMAND":100,
            "data_dir":dir_start+'scenario_3.json',
        },
        {
            "ID": "Scen4_HF",
            "Probability": 1/8,
            "DEMAND":200,
            "data_dir":dir_start+'scenario_4.json',
        },
        {
            "ID": "Scen5_HF",
            "Probability": 1/8,
            "DEMAND":100,
            "data_dir":dir_start+'scenario_5.json',
        },
        {
            "ID": "Scen6_HF",
            "Probability": 1/8,
            "DEMAND":200,
            "data_dir":dir_start+'scenario_6.json',
        },
        {
            "ID": "Scen7_HF",
            "Probability": 1/8,
            "DEMAND":100,
            "data_dir":dir_start+'scenario_7.json',
        },
        {
            "ID": "Scen8_HF",
            "Probability": 1/8,
            "DEMAND":200,
            "data_dir":dir_start+'scenario_8.json',
        }
        ]
    },
}
def HF_PH():
    print("-" * 60)
    print("Running HF_PH")
    print("-" * 60)

    sp = stochastic_program(first_stage_variables=["ug[*,*]"])
    sp.initialize_model(
        name="HF",
        model_data=model_data_noisey["HF"],
        model_builder=HF_builder,
    )

    solver = ProgressiveHedgingSolver_MPISPPY()
    solver.set_options(
        solver="gurobi",
        max_iterations=20,
        loglevel="INFO",
        mpisppy_options=[
            "--lagrangian",
            "--xhatshuffle",
            "--rel-gap=0.01",
            "--default-rho=100",
        ],
    )

    results_mpi = solver.solve(sp, solver="gurobi")

    if getattr(solver, "mpi_rank", 0) == 0:
        pprint.pprint(results_mpi.to_dict())


if __name__ == "__main__":
    HF_PH()