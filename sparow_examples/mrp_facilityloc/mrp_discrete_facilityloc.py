import pyomo.environ as pyo
import itertools
import math
import random
import numpy as np
from pathlib import Path
from sparow.sp import stochastic_program
from sparow.ci import CIProblemAdapter

"""
FACILITY LOCATION
    - HF model is a MIP (first-stage binary variables, second-stage BINARY AND CONTINUOUS variables)
        - HF scenarios can be Low, Medium, and High for each customer (e.g., ['Low', 'High', 'Low', 'Medium'])
        - Each facility can support a fixed number of customers (binary vars in the second stage)
    - LF model relaxes the second stage binary vars
        - LF scenarios are the same as HF
        - Constraint ensuring logic between z and y is taken out so that y takes on continuous values
* Can specify number of scenarios by including "num_HF" key in app_data; otherwise, defaults to 8
* Problem data adapted from https://ampl.com/colab/notebooks/ampl-development-tutorial-26-stochastic-capacitated-facility-location-problem.html#problem-description
"""

app_data = {"n": 3, "t": 4}  # number of facilities & customers
app_data["f"] = [400000, 200000, 600000]  # fixed costs for opening facilities
app_data["c"] = [
    [5739.725, 6539.725, 8650.40, 22372.1125],
    [6055.05, 6739.055, 8050.40, 21014.225],
    [8650.40, 7539.055, 4539.72, 15024.325],
]  # servicing costs
app_data["k"] = [1550, 650, 1750]  # facility capacity
app_data["num_HF"] = 10
app_data["s"] = [1, 1, 2] # max number of customers each facility can service
app_data["a"] = [
    [5688.12, 6601.44, 8723.91, 21998.50],
    [6110.38, 6682.17, 7962.35, 21280.77],
    [8581.33, 7604.28, 4582.91, 14811.46],
]  # transportation costs

BASE_DIR = Path(__file__).resolve().parent # path to directory that contains this file

bigM_path = BASE_DIR / "bigM.txt"
with open(bigM_path, "r") as file: # read in big-M value from bigM.txt
    bigM_str = file.read()
app_data["bigM"] = float(bigM_str)

scens_path = BASE_DIR / "scens_list.npy"
scenarios = np.load(scens_path, allow_pickle=True).tolist()
model_data = {
    "data": {}, # deterministic, model-specific parameters not already in app_data 
    "scenarios": scenarios, # list of scenario dictionaries, each containing at least an "ID" plus the scenario-specific data
}

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
    model.y = pyo.Var(model.N, model.T, domain=[0,1]) # y[i, j] in [0,1] if customer j's demand is met by facility i (RELAXED VAR)
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
    #def LogicFacilities_rule(model, i, j):
    #    return model.z[i, j] <= bigM*model.y[i, j]
        # if facility i doesn't meet customer j's demand, volume of demand met by i for j is 0
    #model.LogicFacilities = pyo.Constraint(model.N, model.T, rule=LogicFacilities_rule)

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
    model.y = pyo.Var(model.N, model.T, within=pyo.Binary) # y[i, j] == 1 if customer j's demand is met by facility i
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
        return model.z[i, j] <= bigM*model.y[i, j]
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


#
# options to solve LF and HF models:
#


def HF_mrp_discrete_facilityloc():
    print("\n Initializing HF MRP discrete facilityloc model...")
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data, model_builder=HF_builder
    )
    return sp


def LF_mrp_discrete_facilityloc():
    print("\n Initializing LF MRP discrete facilityloc model...")
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="LF", model_data=model_data, model_builder=LF_builder
    )
    return sp


# ==== CI ADAPTER ==============================================================

class FacilityLocCIAdapter(CIProblemAdapter):
    """
    Adapter that makes HF and LF facility location models compatible with
    generic sparow.ci ACVMRP / true-gap evaluation code for estimating confidence intervals.

    This class implements the 4 abstract methods required by the
    core sparow.ci CIProblemAdapter base class:
        1. get_scenario_population()
        2. build_model_data(scenarios)
        3. build_stochastic_program(model_data) [THIS IS THE HF MODEL]
        4. first_stage_variable_order()

    It also implements a required_scenario_keys() method that is specific to this problem's data.

    Moreover, it overrides the following methods in order to support low-fidelity models for ACV-MRP:
        - build_low_fidelity_stochastic_program(model_data)
        - get_fidelity_levels()  # returns ["high", "low"]
        - supports_acv()  # returns True
    """

    def __init__(
        self,
        model_name,
        scenario_data,
        model_builder,
        app_data=None,
        first_stage_variables=None,
    ):
        self.model_name = model_name
        self.scenario_data = scenario_data
        self.model_builder = model_builder
        self.app_data = {} if app_data is None else dict(app_data)
        self.first_stage_variables = (
            ["x"]
            if first_stage_variables is None
            else first_stage_variables
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
        Build and return the Sparow stochastic_program object for this
        facility location model instance. THIS IS THE HF MODEL.
        """
        sp = stochastic_program(first_stage_variables=self.first_stage_variables)
        sp.initialize_application(app_data=self.app_data)
        sp.initialize_model(
            name="HF",
            model_data=model_data,
            model_builder=HF_builder,
        )
        return sp

    def first_stage_variable_order(self):
        """
        Return the ordered list of first-stage variable names.

        This order is used by the generic CI code to:
            - extract xhat from solved EF results,
            - convert xhat dicts into vectors for sp.evaluate(...).
        """
        # Return the first-stage variables in order: x[0], x[1], x[2], ...
        n = self.app_data.get("n", 3)
        return [f"x[{i}]" for i in range(n)]

    def required_scenario_keys(self):
        """
        Facility location scenarios dictionaries must contain a Demand field in addition to the
        always-required "ID" and "Probability" keys.
        """
        return ["Demand"]

    def build_low_fidelity_stochastic_program(self, model_data):
        """
        Build and return the low-fidelity stochastic program for ACV-MRP.
        This is the LF (relaxed) model where second-stage binary variables are continuous.
        """
        sp = stochastic_program(first_stage_variables=self.first_stage_variables)
        sp.initialize_application(app_data=self.app_data)
        sp.initialize_model(
            name="LF",
            model_data=model_data,
            model_builder=LF_builder,
        )
        return sp

    def get_fidelity_levels(self):
        """Return list of supported fidelity levels."""
        return ['high', 'low']

    def supports_acv(self):
        """
        Whether this adapter supports ACV-MRP.
        Returns True since we have both HF and LF models implemented.
        """
        return True


# =================================================================
# Core CI code expects exactly one standard factory name
# =================================================================

def get_ci_problem_adapter(model_name="HF", use_integer=False):
    """
    Module-level factory function expected by the generic sparow.ci core code.

    This function dispatches to the appropriate facility location-specific CI adapter
    (HF or LF) based on `model_name`, while exposing one standard
    factory name that the core CI logic expects to call.
    """
    if model_name == "HF":
        return get_hf_ci_problem_adapter()
    if model_name == "LF":
        return get_lf_ci_problem_adapter()
    raise ValueError(f"Unknown facility location model_name: {model_name}")


def get_hf_ci_problem_adapter():
    return FacilityLocCIAdapter(
        model_name="HF",
        scenario_data=model_data,
        model_builder=HF_builder,
        app_data=app_data,
        first_stage_variables=["x"],
    )


def get_lf_ci_problem_adapter():
    return FacilityLocCIAdapter(
        model_name="LF",
        scenario_data=model_data,
        model_builder=LF_builder,
        app_data=app_data,
        first_stage_variables=["x"],
    )
