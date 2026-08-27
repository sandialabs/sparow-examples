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
