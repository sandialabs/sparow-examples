from sparow.sp import stochastic_program
from sparow.ef import ExtensiveFormSolver
from scenarios import build_noisy_hf_scenarios
from model_builders import HF_builder, LF_builder_SHED

dir_start='data/rts_noisey/scenarios_2/'
dir_start='data/rts_noisey/MARCH_300/'

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

        ]
    },
}



def main():
    print("-" * 60)
    print("Running HF_EF")
    print("-" * 60)

    #model_data_noisey = build_noisy_hf_scenarios('data/rts_noisey/scenarios_2/', n_scenarios=8)

    sp = stochastic_program(first_stage_variables=["ug[*,*]"])
    sp.initialize_model(
        name="HF",
        model_data=model_data_noisey["HF"],
        model_builder=LF_builder_SHED,
    )

    solver = ExtensiveFormSolver()
    solver.set_options(solver="gurobi", loglevel="INFO")
    results = solver.solve(sp)
    results.write("result_jsons/results_HF_EF.json", indent=4)

if __name__ == "__main__":
    main()
