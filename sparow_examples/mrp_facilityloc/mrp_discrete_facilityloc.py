import pyomo.environ as pyo
import itertools
import math
import random
import numpy as np
from pathlib import Path

from sparow.sp import stochastic_program
from sparow.conf_intervals import CIProblemAdapter

import argparse
import json
import os

"""
FACILITY LOCATION
    - HF model is a MIP (first-stage binary variables, second-stage BINARY AND CONTINUOUS variables)
        - Each facility can support a fixed number of customers (binary vars in the second stage)
    - LF model relaxes the second stage binary vars
        - LF scenarios are the same as HF
        - Constraint ensuring logic between z and y is taken out so that y takes on continuous values
* Scenarios are generated using linear interpolation between low and high demand values for each city, 
  creating a discrete uniform distribution over all resulting scenarios
* The number of scenarios is controlled by the shell script's N parameter, not by app_data
* You can run this file as a script to write the full scenario population to a .json or .npy file
"""

app_data = {"n": 6, "t": 4}  # number of facilities & customers
app_data["f"] = [
    260000,
    275000,
    270000,
    285000,
    320000,
    340000,
]  # fixed costs for opening facilities
app_data["c"] = [
    [4200.0, 5200.0, 12500.0, 18000.0],  # facility 0 good for cust 0,1
    [4600.0, 4800.0, 11800.0, 17500.0],  # facility 1 also good for cust 0,1
    [12800.0, 12000.0, 4100.0, 5600.0],  # facility 2 good for cust 2,3
    [13500.0, 12600.0, 4500.0, 5100.0],  # facility 3 also good for cust 2,3
    [7600.0, 7900.0, 7800.0, 8200.0],  # facility 4 compromise facility
    [9000.0, 9400.0, 9100.0, 9600.0],  # facility 5 dominated-ish but feasible
]  # servicing costs
app_data["k"] = [2500, 2500, 2500, 2500, 2500, 2500]  # facility capacity
app_data["s"] = [2, 2, 2, 2, 2, 2]  # max number of customers each facility can service
app_data["a"] = [
    [3900.0, 5000.0, 12000.0, 17000.0],
    [4300.0, 4700.0, 11400.0, 16800.0],
    [12200.0, 11600.0, 3900.0, 5200.0],
    [12900.0, 12100.0, 4300.0, 4800.0],
    [7000.0, 7300.0, 7100.0, 7600.0],
    [8600.0, 9000.0, 8700.0, 9200.0],
]  # transportation costs

BASE_DIR = Path(__file__).resolve().parent  # path to directory that contains this file

bigM_path = BASE_DIR / "bigM.txt"
with open(bigM_path, "r") as file:  # read in big-M value from bigM.txt
    bigM_str = file.read()
app_data["bigM"] = float(bigM_str)


# ==== SCENARIO DATA ===========================================================

# Define low and high demand values for each city (customer)
low_demands = [180.0, 500.0, 140.0, 40.0]
high_demands = [700.0, 1400.0, 650.0, 260.0]


class FacilityLocationScenarioData(object):
    """
    Construct the full finite population set of facility-location scenarios.

    For each customer, linearly interpolate between the low and high demand
    values using num_data_points support points. The full
    scenario distribution is then the Cartesian product of those support
    points across the customers, with equal probability assigned to each
    possible scenario vector.
    """

    def __init__(self, num_data_points):
        self.num_data_points = num_data_points

        self.demand_supports = []
        for low, high in zip(low_demands, high_demands):
            self.demand_supports.append(np.linspace(low, high, num_data_points))

    def scenario_generator(self):
        """
        Final output is a dictionary with a single key-value pair.
        The key is "scenarios"
        The value is a list, called scen_dict_list. It contains one dictionary per possible
        population scenario. Each scenario's dictionary must contain "ID", demands, and
        "Probability".
        """
        total_scens = self.num_data_points ** len(self.demand_supports)
        scen_prob = 1.0 / total_scens  # each scenario vector gets equal probability

        scen_id = 0  # naming convention: each scenario ID string ends in a number (population index)
        scen_dict_list = []

        # Use itertools.product to get cartesian product
        for demand_tuple in itertools.product(*self.demand_supports):
            scen_dict_list.append(
                {
                    "ID": f"scen_{scen_id}",
                    "Demand": [float(d) for d in demand_tuple],
                    "Probability": scen_prob,
                }
            )
            scen_id += 1

        return {"scenarios": scen_dict_list}


HFScenarioObject = FacilityLocationScenarioData(num_data_points=5)
LFScenarioObject = FacilityLocationScenarioData(num_data_points=5)

HF_scendata = HFScenarioObject.scenario_generator()
LF_scendata = LFScenarioObject.scenario_generator()


# ==== MODEL DATA ===============================================================

# This is a multi-model container:
# stores scenario datasets for each model
scenario_data_by_model = {
    "HF": HF_scendata,
    "LF": LF_scendata,
}

# ==== MODEL BUILDERS ===========================================================


def LF_builder(data, args):
    n = data["n"]
    t = data["t"]
    f = data["f"]
    c = data["c"]
    k = data["k"]
    bigM = data["bigM"]
    s = data["s"]
    a = data["a"]

    ### STOCHASTIC DATA
    d = data["Demand"]

    model = pyo.ConcreteModel(data["ID"])

    ### PARAMETERS
    model.N = pyo.Set(initialize=[i for i in range(n)])
    model.T = pyo.Set(initialize=[j for j in range(t)])

    ### VARIABLES
    model.x = pyo.Var(model.N, within=pyo.Binary)  # x[i] == 1 if facility i is open
    model.y = pyo.Var(
        model.N, model.T, domain=[0, 1]
    )  # y[i, j] in [0,1] if customer j's demand is met by facility i (RELAXED VAR)
    model.z = pyo.Var(
        model.N, model.T, within=pyo.NonNegativeReals
    )  # z[i, j] = volume of customer j's demand met by facility i

    ### CONSTRAINTS
    def MeetDemand_rule(model, j):
        return sum(model.z[i, j] for i in range(n)) >= d[j]
        # sum of demand met by all facilities for customer j is greater than demand from customer j

    model.MeetDemand = pyo.Constraint(model.T, rule=MeetDemand_rule)

    def SufficientProduction_rule(model):
        return sum(k[i] * model.x[i] for i in range(n)) >= sum(d[j] for j in range(t))
        # sum of production from all facilities is greater than sum of total demand from all customers

    model.SufficientProduction = pyo.Constraint(rule=SufficientProduction_rule)

    def Capacity_rule(model, i):
        return sum(model.z[i, j] for j in range(t)) <= k[i] * model.x[i]
        # volume of demand met is less than capacity for each facility. this constraint also ensures logic between x, z

    model.Capacity = pyo.Constraint(model.N, rule=Capacity_rule)

    def OpenFacilities_rule(model, i, j):
        return model.y[i, j] <= model.x[i]
        # facility i needs to be open to fulfill customer j's demand w/ facility i

    model.OpenFacilities = pyo.Constraint(model.N, model.T, rule=OpenFacilities_rule)

    ### COMMENTING THIS OUT SO THAT y TAKES ON CONTINUOUS VALUES ###
    # def LogicFacilities_rule(model, i, j):
    #    return model.z[i, j] <= bigM*model.y[i, j]
    # if facility i doesn't meet customer j's demand, volume of demand met by i for j is 0
    # model.LogicFacilities = pyo.Constraint(model.N, model.T, rule=LogicFacilities_rule)

    def CustomersPerFacility_rule(model, i):
        return sum(model.y[i, j] for j in range(t)) <= s[i]
        # limit on the number of customers serviced by facility j

    model.CustomersPerFacility = pyo.Constraint(model.N, rule=CustomersPerFacility_rule)

    ### OBJECTIVE
    def Obj_rule(model):
        # cost of fulfilling total volume of demand
        expr = sum(sum(c[i][j] * model.z[i, j] for j in range(t)) for i in range(n))
        # cost of transporting goods from facility i to customer j
        expr += sum(sum(a[i][j] * model.y[i, j] for j in range(t)) for i in range(n))
        # cost of opening facilities
        expr += sum(f[i] * model.x[i] for i in range(n))
        return expr

    model.obj = pyo.Objective(rule=Obj_rule, sense=pyo.minimize)

    return model


def HF_builder(data, args):
    n = data["n"]
    t = data["t"]
    f = data["f"]
    c = data["c"]
    k = data["k"]
    bigM = data["bigM"]
    s = data["s"]
    a = data["a"]

    ### STOCHASTIC DATA
    d = data["Demand"]

    model = pyo.ConcreteModel(data["ID"])

    ### PARAMETERS
    model.N = pyo.Set(initialize=[i for i in range(n)])
    model.T = pyo.Set(initialize=[j for j in range(t)])

    ### VARIABLES
    model.x = pyo.Var(model.N, within=pyo.Binary)  # x[i] == 1 if facility i is open
    model.y = pyo.Var(
        model.N, model.T, within=pyo.Binary
    )  # y[i, j] == 1 if customer j's demand is met by facility i
    model.z = pyo.Var(
        model.N, model.T, within=pyo.NonNegativeReals
    )  # z[i, j] = volume of customer j's demand met by facility i

    ### CONSTRAINTS
    def MeetDemand_rule(model, j):
        return sum(model.z[i, j] for i in range(n)) >= d[j]
        # sum of demand met by all facilities for customer j is greater than demand from customer j

    model.MeetDemand = pyo.Constraint(model.T, rule=MeetDemand_rule)

    def SufficientProduction_rule(model):
        return sum(k[i] * model.x[i] for i in range(n)) >= sum(d[j] for j in range(t))
        # sum of production from all facilities is greater than sum of total demand from all customers

    model.SufficientProduction = pyo.Constraint(rule=SufficientProduction_rule)

    def Capacity_rule(model, i):
        return sum(model.z[i, j] for j in range(t)) <= k[i] * model.x[i]
        # volume of demand met is less than capacity for each facility. this constraint also ensures logic between x, z

    model.Capacity = pyo.Constraint(model.N, rule=Capacity_rule)

    def OpenFacilities_rule(model, i, j):
        return model.y[i, j] <= model.x[i]
        # facility i needs to be open to fulfill customer j's demand w/ facility i

    model.OpenFacilities = pyo.Constraint(model.N, model.T, rule=OpenFacilities_rule)

    def LogicFacilities_rule(model, i, j):
        return model.z[i, j] <= bigM * model.y[i, j]
        # if facility i doesn't meet customer j's demand, volume of demand met by i for j is 0

    model.LogicFacilities = pyo.Constraint(model.N, model.T, rule=LogicFacilities_rule)

    def CustomersPerFacility_rule(model, i):
        return sum(model.y[i, j] for j in range(t)) <= s[i]
        # limit on the number of customers serviced by facility j

    model.CustomersPerFacility = pyo.Constraint(model.N, rule=CustomersPerFacility_rule)

    ### OBJECTIVE
    def Obj_rule(model):
        # cost of fulfilling total volume of demand
        expr = sum(sum(c[i][j] * model.z[i, j] for j in range(t)) for i in range(n))
        # cost of transporting goods from facility i to customer j
        expr += sum(sum(a[i][j] * model.y[i, j] for j in range(t)) for i in range(n))
        # cost of opening facilities
        expr += sum(f[i] * model.x[i] for i in range(n))
        return expr

    model.obj = pyo.Objective(rule=Obj_rule, sense=pyo.minimize)

    return model


# ==== CI ADAPTER ==============================================================


class FacilityLocCIAdapter(CIProblemAdapter):
    """
    Adapter that makes HF and LF facility location models compatible with
    generic sparow.ci ACVMRP / true-gap evaluation code for estimating confidence intervals.

    This class implements the 4 abstract methods required by the
    core sparow.ci CIProblemAdapter base class:
        1. get_scenario_population()
        2. build_model_data(scenarios)
        3. build_stochastic_program(model_data) [DEFAULTS TO HF MODEL]
        4. first_stage_variable_order()

    It also implements a required_scenario_keys() method that is specific to this problem's data.

    Moreover, it overrides the following methods in order to support low-fidelity models for ACV-MRP:

        - get_fidelity_levels()  # returns ["high", "low"]
        - supports_acv()  # returns True
        - set_active_fidelity
        - get_active_fidelity
    """

    def __init__(
        self,
        model_name,
        scenario_data,
        model_builder,
        app_data=None,
        first_stage_variables=None,
    ):
        super().__init__(
            model_name=model_name,
            scenario_data=scenario_data,
            model_builder=model_builder,
            app_data=app_data,
            first_stage_variables=(
                ["x"] if first_stage_variables is None else first_stage_variables
            ),
        )
        if model_name == "HF":
            self._active_fidelity = "high"
        elif model_name == "LF":
            self._active_fidelity = "low"
        else:
            raise RuntimeError(
                f"Unrecognized model_name for discrete facilityloc: {model_name}"
            )

    def get_scenario_population(self):
        """
        Return the full finite / historical scenario population as a list
        of scenario dictionaries.
        """
        return self.scenario_data["scenarios"]

    def build_model_data(self, scenarios):
        """
        Build the model_data dictionary expected by Sparow.
        """
        return {"data": {}, "scenarios": scenarios}

    def build_stochastic_program(self, model_data):
        """
        Build and return the stochastic_program object for the currently active fidelity.
        """
        # print(f"Active fidelity state: {self._active_fidelity}")
        sp = stochastic_program(first_stage_variables=self.first_stage_variables)
        sp.initialize_application(app_data=self.app_data)

        if self._active_fidelity == "high":
            # print("Initializing HF model")
            sp.initialize_model(
                name="HF",
                model_data=model_data,
                model_builder=HF_builder,
            )
        elif self._active_fidelity == "low":
            # print("Initializing LF model")
            sp.initialize_model(
                name="LF",
                model_data=model_data,
                model_builder=LF_builder,
            )
        else:
            raise RuntimeError(f"Invalid active fidelity: {self._active_fidelity}")

        return sp

    def first_stage_variable_order(self):
        """
        Return the ordered list of first-stage variable names.

        This order is used by the generic CI code to:
            - extract xhat from solved EF results,
            - convert xhat dicts into vectors for sp.evaluate(...).
        """
        # Return the first-stage variables in order: x[0], x[1], x[2], ...
        n = self.app_data.get(
            "n", 3
        )  # 3 is the default number of facilities if not specified in app_data
        return [f"x[{i}]" for i in range(n)]

    def required_scenario_keys(self):
        """
        Facility location scenarios dictionaries must contain a Demand field in addition to the
        always-required "ID" and "Probability" keys.
        """
        return ["Demand"]

    def get_fidelity_levels(self):
        """Return list of supported fidelity levels."""
        return ["high", "low"]

    def supports_acv(self):
        """
        Whether this adapter supports ACV-MRP.
        Returns True since we have both HF and LF models implemented.
        """
        return True

    def set_active_fidelity(self, fidelity):
        """
        Set the active fidelity level used by build_stochastic_program().
        """
        if fidelity not in ("high", "low"):
            raise ValueError(f"Unknown fidelity level: {fidelity}")
        self._active_fidelity = fidelity

    def get_active_fidelity(self):
        return self._active_fidelity

    def scenario_vector_keys(self):
        """
        Return the scenario dictionary keys that correspond to uncertain problem data
        """
        return ["Demand"]

    def decode_scenario_vector(self, vector, scenario_id: str):
        """
        Rebuild a facility-location scenario dictionary in the
        correct format from a flat numeric vector.

        NOTE: probability key and value is computed in the internal
        PyApproxModelWrapper logic.
        """
        return {"ID": scenario_id, "Demand": [float(elem) for elem in vector]}


# =================================================================
# Core CI code expects exactly one standard factory name
# =================================================================


def get_ci_problem_adapter(model_name="HF", use_integer=False, lf_model_type="classic"):
    """
    Module-level factory function expected by the generic sparow.ci core code.

    This function dispatches to the appropriate facility location-specific CI adapter
    (HF or LF) based on "model_name" argument.

    NOTE: lf_model_type is dummy argument here
    """
    if model_name == "HF":
        # print("Returning HF Problem Adapter")
        return get_hf_ci_problem_adapter()
    if model_name == "LF":
        # print("Returning LF Problem Adapter")
        return get_lf_ci_problem_adapter()
    raise ValueError(f"Unknown facility location model_name: {model_name}")


def get_hf_ci_problem_adapter():
    # print("\nBuilding HF Discrete FacilityLoc\n")
    return FacilityLocCIAdapter(
        model_name="HF",
        scenario_data=scenario_data_by_model["HF"],
        model_builder=HF_builder,
        app_data=app_data,
        first_stage_variables=["x"],
    )


def get_lf_ci_problem_adapter():
    # print("\nBuilding LF Discrete FacilityLoc\n")
    return FacilityLocCIAdapter(
        model_name="LF",
        scenario_data=scenario_data_by_model["LF"],
        model_builder=LF_builder,
        app_data=app_data,
        first_stage_variables=["x"],
    )


# =================================================================
# Write the scenario data to file for use in the CI tests
# =================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Write the full facility location scenario population to a file "
        "that sparow.ci.cli --scenario-file can read."
    )
    parser.add_argument(
        "--output", required=True, help="Output file path ending in .json or .npy"
    )
    parser.add_argument(
        "--num-data-points",
        type=int,
        default=10,
        help="Number of interpolated demand values per customer (default: 10)",
    )
    args = parser.parse_args()

    scenario_object = FacilityLocationScenarioData(args.num_data_points)
    scenario_data = scenario_object.scenario_generator()
    scen_dict_list = scenario_data["scenarios"]
    print(f"\n ==== Number of population scenarios: {len(scen_dict_list)} === \n")

    # Scenarios are the same for the LF and HF models
    adapter = FacilityLocCIAdapter(
        model_name="HF",
        scenario_data=scenario_data,
        model_builder=HF_builder,
        app_data=app_data,
        first_stage_variables=["x"],
    )

    scenarios = adapter.get_scenario_population()
    adapter.validate_scenario_population(scenarios)

    outpath = os.path.abspath(args.output)
    (
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        if os.path.dirname(outpath)
        else None
    )

    if outpath.endswith(".json"):
        with open(outpath, "w") as f:
            json.dump({"scenarios": scenarios}, f, indent=2)
    elif outpath.endswith(".npy"):
        np.save(outpath, {"scenarios": scenarios}, allow_pickle=True)
    else:
        raise ValueError("Output file must end with .json or .npy")

    print(f"Wrote {len(scenarios)} scenarios to: {outpath}")
    print(f"Use this with: --scenario-file {outpath}")


if __name__ == "__main__":
    main()
