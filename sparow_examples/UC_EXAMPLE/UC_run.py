import os
import time

from sparow.sp import stochastic_program
from sparow.ef import ExtensiveFormSolver

import egret.parsers.prescient_dat_parser as pdp
import egret.data.model_data as md
import egret.models.unit_commitment as uc
import mpisppy.utils.sputils as sputils

from pyomo.dataportal import DataPortal
from pyomo.environ import value, SolverFactory
from IPython import embed

def uc_model_builder(data, args):
    path = data["data_dir"]
    scenario_name = data["ID"]

    scennum = sputils.extract_num(scenario_name)

    uc_model_params = pdp.get_uc_model()

    scenario_data = DataPortal(model=uc_model_params)
    scenario_data.load(filename=os.path.join(path, "RootNode.dat"))
    scenario_data.load(filename=os.path.join(path, f"Node{scennum}.dat"))

    scenario_params = uc_model_params.create_instance(
        scenario_data,
        report_timing=False,
        name=scenario_name,
    )

    scenario_md = md.ModelData(
        pdp.create_model_data_dict_params(scenario_params, keep_names=True)
    )

    scenario_instance = uc.create_tight_unit_commitment_model(
        scenario_md,
        network_constraints="power_balance_constraints",
    )

    return scenario_instance


if __name__ == "__main__":
    scen_count = 10
    scenario_names = [f"Scenario{i+1}" for i in range(scen_count)]
    probability = 1.0 / scen_count

    model_data_uc = {
        "scenarios": [
            {
                "ID": scenario,
                "Probability": probability,
                "data_dir": "/home/zakilwe/AGM/AGMDecentralized/mpi-sppy/examples/uc/10scenarios_r1/",
            }
            for scenario in scenario_names
        ]
    }

    # You must choose first-stage variables that are nonanticipative
    # for the UC model. These depend on the variable names in the Egret model.
    sp = stochastic_program(
        first_stage_variables=[
            "UnitOn[*,*]",
        ]
    )

    sp.initialize_model(
        name="UC",
        model_data=model_data_uc,
        model_builder=uc_model_builder,
    )

    solver = ExtensiveFormSolver()
    solver.set_options(solver="gurobi", loglevel="INFO")

    M = sp.create_EF(compact_repn=False)

    opt = SolverFactory("gurobi")

    start = time.time()
    results = opt.solve(M, tee=True)
    end = time.time()

    #start = time.time()
    #res = solver.solve_and_return_EF(sp)
    #end = time.time()

    print(f"Elapsed time: {end - start:.4f} seconds")
