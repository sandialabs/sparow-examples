import random

def build_model_data_time(scenarios, bound, data_dir, seed=55):
    """
    Builds finite population of scenarios that we can sample from.
    """
    random.seed(seed)
    probability = 1 / len(scenarios)

    scenario_list = []

    for scenario in scenarios:
        scenario_list.append(
        {
            "ID": scenario,
            "Probability": probability,
            "DEMAND_1": 1.0,
            "DEMAND_2": 1.0 + random.uniform(-bound, bound),
            "DEMAND_3": 1.0 + random.uniform(-bound, bound),
            "DEMAND_4": 1.0 + random.uniform(-bound, bound),
            "data_dir": data_dir,
        }
    )

    return {"scenarios": scenario_list}

# def build_model_data_time(scenarios, bound, data_dir, seed=55):
#     random.seed(seed)
#     probability = 1 / len(scenarios)

#     return {
#         'HF': {
#             "scenarios": [
#                 {
#                     "ID": scenario,
#                     "Probability": probability,
#                     "DEMAND_1": 1.0,
#                     "DEMAND_2": 1.0 + random.uniform(-bound, bound),
#                     "DEMAND_3": 1.0 + random.uniform(-bound, bound),
#                     "DEMAND_4": 1.0 + random.uniform(-bound, bound),
#                     "data_dir": data_dir,
#                 }
#                 for scenario in scenarios
#             ]
#         }
#     }

# def split_multifidelity(model_data_time):
#     scenarios = model_data_time['HF']["scenarios"]
#     return {
#         'HF': {"scenarios": scenarios[:4]},
#         'LF': {"scenarios": scenarios[4:]},
#     }
