import pyomo.environ as pyo
import itertools
import math
import random
from sparow.sp import stochastic_program

"""
FACILITY LOCATION
    - HF model is a MIP (first-stage binary variables, second-stage continuous variables)
        - HF scenarios can be Low, Medium, and High for each customer (e.g., ['Low', 'High', 'Low', 'Medium'])
    - LF1 model is also a MIP, scenarios chosen from same options
        - Approximates capacity constraint with big-M
    - LF2 model is also a MIP, scenarios chosen from same options
        - Approximates the demand constraint via aggregation
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

customer_demand = {
    "San_Antonio_TX": [450, 650, 887],
    "Dallas_TX": [910, 1134, 1456],
    "Jackson_MS": [379, 416, 673],
    "Birmingham_AL": [91, 113, 207],
}
app_data["bigM"] = max(max(val) for val in customer_demand.values())

demand_levels = ["Low", "Medium", "High"]
cities = list(customer_demand.keys())
all_scenarios = list(itertools.product(demand_levels, repeat=len(cities)))

# mapping of demand levels to their corresponding values
demand_value_mapping = {"Low": 0, "Medium": 1, "High": 2}
# mapping of demand levels to their corresponding probabilities
demand_prob_mapping = {"Low": 0.25, "Medium": 0.5, "High": 0.25}

sdict = {}  # dictionary of all possible Low/Medium/High combinations for HF scenarios
for scenario in all_scenarios:
    scenario_dict = dict(zip(cities, scenario))
    sdict[scenario] = {
        "Demand": {
            city: customer_demand[city][demand_value_mapping[demand]]
            for city, demand in scenario_dict.items()
        },
        "Probability": math.prod(
            demand_prob_mapping[scenario_dict[city]] for city in scenario_dict.keys()
        ),
    }


def scenarios_to_scens_list(scenario_dict, demand_dict, value_mapping_dict, seed, app_data):
    random.seed(seed)
    scenarios = random.choices(
        list(scenario_dict.keys()), k=app_data.get("num_scenarios", 8)
    )  # randomly select scenarios from scenario_dict
    scens_list = []  # list of scenarios
    customer_demand_vals = list(demand_dict.values())
    for scen in scenarios:
        scens_list.append(
            {
                "ID": f"{scen}",
                "Demand": [
                    customer_demand_vals[customer][value_mapping_dict[scen[customer]]]
                    for customer in range(len(scen))
                ],
                "Probability": scenario_dict[scenario]["Probability"],
            }
        )

    # normalize HF scenario probabilities
    norm_term = sum(
        scens_list[s_idx]["Probability"] for s_idx in range(len(scens_list))
    )
    for s_idx in range(len(scens_list)):
        scens_list[s_idx]["Probability"] /= norm_term

    return scens_list


LF1_scens_list = scenarios_to_scens_list(
    scenario_dict=sdict,
    demand_dict=customer_demand,
    value_mapping_dict=demand_value_mapping,
    seed=98765432123456789,
    app_data=app_data
)
LF2_scens_list = scenarios_to_scens_list(
    scenario_dict=sdict,
    demand_dict=customer_demand,
    value_mapping_dict=demand_value_mapping,
    seed=12345678987654321,
    app_data=app_data
)
HF_scens_list = scenarios_to_scens_list(
    scenario_dict=sdict,
    demand_dict=customer_demand,
    value_mapping_dict=demand_value_mapping,
    seed=58564564871312356,
    app_data=app_data
)

model_data = {
    "LF1": {"scenarios": LF1_scens_list},
    "LF2": {"scenarios": LF2_scens_list},
    "HF": {"scenarios": HF_scens_list},
}


def LF1_builder(data, args):
    n = data["n"]
    t = data["t"]
    f = data["f"]
    c = data["c"]
    k = data["k"]
    bigM = data["bigM"]

    ### STOCHASTIC DATA
    d = data["Demand"]

    model = pyo.ConcreteModel(data["ID"])

    ### PARAMETERS
    model.N = pyo.Set(initialize=[i for i in range(n)])
    model.T = pyo.Set(initialize=[j for j in range(t)])

    ### VARIABLES
    model.x = pyo.Var(model.N, within=pyo.Binary)  # x[i] == 1 if facility i is open
    model.z = pyo.Var(
        model.N, model.T, within=pyo.NonNegativeReals
    )  # z[i, j] = volume of customer j's demand met by facility i

    ### CONSTRAINTS
    def MeetDemand_rule(model, j):
        return sum(model.z[i, j] for i in range(n)) >= d[j]

    model.MeetDemand = pyo.Constraint(model.T, rule=MeetDemand_rule)

    def SufficientProduction_rule(model):
        return sum(k[i] * model.x[i] for i in range(n)) >= sum(d[j] for j in range(t))

    model.SufficientProduction = pyo.Constraint(rule=SufficientProduction_rule)

    def Logic_rule(
        model, i
    ):  # relaxation of capacity constraint that still enforces logic between x, z
        return sum(model.z[i, j] for j in range(t)) <= bigM * model.x[i]

    model.Logic = pyo.Constraint(model.N, rule=Logic_rule)

    ### OBJECTIVE
    def Obj_rule(model):
        expr = sum(sum(c[i][j] * model.z[i, j] for j in range(t)) for i in range(n))
        expr += sum(f[i] * model.x[i] for i in range(n))
        return expr

    model.obj = pyo.Objective(rule=Obj_rule, sense=pyo.minimize)

    return model


def LF2_builder(data, args):
    n = data["n"]
    t = data["t"]
    f = data["f"]
    c = data["c"]
    k = data["k"]

    ### STOCHASTIC DATA
    d = data["Demand"]

    model = pyo.ConcreteModel(data["ID"])

    ### PARAMETERS
    model.N = pyo.Set(initialize=[i for i in range(n)])
    model.T = pyo.Set(initialize=[j for j in range(t)])

    ### VARIABLES
    model.x = pyo.Var(model.N, within=pyo.Binary)  # x[i] == 1 if facility i is open
    model.z = pyo.Var(
        model.N, model.T, within=pyo.NonNegativeReals
    )  # z[i, j] = volume of customer j's demand met by facility i

    ### CONSTRAINTS
    def AggDemand_rule(model):  # approximation (aggregation) of demand constraints
        return sum(sum(model.z[i, j] for i in range(n)) for j in range(t)) >= sum(
            d[j] for j in range(t)
        )

    model.AggDemand = pyo.Constraint(rule=AggDemand_rule)

    def SufficientProduction_rule(model):
        return sum(k[i] * model.x[i] for i in range(n)) >= sum(d[j] for j in range(t))

    model.SufficientProduction = pyo.Constraint(rule=SufficientProduction_rule)

    def Capacity_rule(model, i):  # note this constraint also ensures logic between x, z
        return sum(model.z[i, j] for j in range(t)) <= k[i] * model.x[i]

    model.Capacity = pyo.Constraint(model.N, rule=Capacity_rule)

    ### OBJECTIVE
    def Obj_rule(model):
        expr = sum(sum(c[i][j] * model.z[i, j] for j in range(t)) for i in range(n))
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

    ### STOCHASTIC DATA
    d = data["Demand"]

    model = pyo.ConcreteModel(data["ID"])

    ### PARAMETERS
    model.N = pyo.Set(initialize=[i for i in range(n)])
    model.T = pyo.Set(initialize=[j for j in range(t)])

    ### VARIABLES
    model.x = pyo.Var(model.N, within=pyo.Binary)  # x[i] == 1 if facility i is open
    model.z = pyo.Var(
        model.N, model.T, within=pyo.NonNegativeReals
    )  # z[i, j] = volume of customer j's demand met by facility i

    ### CONSTRAINTS
    def MeetDemand_rule(model, j):
        return sum(model.z[i, j] for i in range(n)) >= d[j]

    model.MeetDemand = pyo.Constraint(model.T, rule=MeetDemand_rule)

    def SufficientProduction_rule(model):
        return sum(k[i] * model.x[i] for i in range(n)) >= sum(d[j] for j in range(t))

    model.SufficientProduction = pyo.Constraint(rule=SufficientProduction_rule)

    def Capacity_rule(model, i):  # note this constraint also ensures logic between x, z
        return sum(model.z[i, j] for j in range(t)) <= k[i] * model.x[i]

    model.Capacity = pyo.Constraint(model.N, rule=Capacity_rule)

    ### OBJECTIVE
    def Obj_rule(model):
        expr = sum(sum(c[i][j] * model.z[i, j] for j in range(t)) for i in range(n))
        expr += sum(f[i] * model.x[i] for i in range(n))
        return expr

    model.obj = pyo.Objective(rule=Obj_rule, sense=pyo.minimize)

    return model


#
# options to solve, LF, HF, or MF random models:
#


def HF_grid_facilityloc():
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )
    return sp


def LF1_grid_facilityloc():
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="LF1", model_data=model_data["LF1"], model_builder=LF1_builder
    )
    return sp


def LF2_grid_facilityloc():
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="LF2", model_data=model_data["LF2"], model_builder=LF2_builder
    )
    return sp


def random_HF_LF1_grid_facilityloc():
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )
    sp.initialize_model(
        name="LF1",
        model_data=model_data["LF1"],
        model_builder=LF1_builder,
        default=False,
    )
    sp.initialize_bundles(
        scheme="mf_random",
        LF=2,
        seed=1234567890,
    )
    return sp


def random_HF_LF2_grid_facilityloc():
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )
    sp.initialize_model(
        name="LF2",
        model_data=model_data["LF2"],
        model_builder=LF2_builder,
        default=False,
    )
    sp.initialize_bundles(
        scheme="mf_random",
        LF=2,
        seed=1234567890,
    )
    return sp


def dissimilar_HF_LF1_grid_facilityloc():
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )
    sp.initialize_model(
        name="LF1",
        model_data=model_data["LF1"],
        model_builder=LF1_builder,
        default=False,
    )
    sp.initialize_bundles(
        scheme="mf_kmeans_dissimilar",
        LF=2,
        seed=1234567890,
    )
    return sp


def dissimilar_HF_LF2_grid_facilityloc():
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )
    sp.initialize_model(
        name="LF2",
        model_data=model_data["LF2"],
        model_builder=LF2_builder,
        default=False,
    )
    sp.initialize_bundles(
        scheme="mf_kmeans_dissimilar",
        LF=2,
        seed=1234567890,
    )
    return sp


def similar_HF_LF1_grid_facilityloc():
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )
    sp.initialize_model(
        name="LF1",
        model_data=model_data["LF1"],
        model_builder=LF1_builder,
        default=False,
    )
    sp.initialize_bundles(
        scheme="mf_kmeans_similar",
        LF=2,
        seed=1234567890,
        data_key="Demand",
    )
    return sp


def similar_HF_LF2_grid_facilityloc():
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )
    sp.initialize_model(
        name="LF2",
        model_data=model_data["LF2"],
        model_builder=LF2_builder,
        default=False,
    )
    sp.initialize_bundles(
        scheme="mf_kmeans_similar",
        LF=2,
        seed=1234567890,
    )
    return sp
