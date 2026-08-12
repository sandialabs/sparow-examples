from config import ExperimentConfig
from scenarios import build_model_data_time, split_multifidelity
from model_builders import LF_builder_multitime_period, HF_builder_multitime_period, uHF_builder_multitime_period
from experiment_runner import run_ef, run_ph
from evaluation import extract_objective, extract_solution_dict, extract_variable_map, compute_percent_error, resolve_on_hf
from reporting import print_summary

def main():
    cfg = ExperimentConfig()
    first_stage_vars = ["time_periods[1].m.pg[*]"]

    model_data_time = build_model_data_time(
        scenarios=cfg.scenarios,
        bound=cfg.bound,
        data_dir=cfg.data_dir,
        seed=cfg.seed,
    )
    model_data_time_MF = split_multifidelity(model_data_time)

    results_summary_EF = {
        "HF_EF": {"cpu_time": None, "objective": None},
        "LF_EF": {"cpu_time": None, "objective": None},
        "MF_EF": {"cpu_time": None, "objective": None},
    }

    # LF_EF
    out = run_ef(
        model_data_dict={"HF": model_data_time["HF"]},
        builder_dict={"HF": HF_builder_multitime_period},
        first_stage_variables=first_stage_vars,
    )
    results_dict_LF = out["results"].to_dict()
    results_summary_EF["LF_EF"]["cpu_time"] = out["cpu_time"]
    results_summary_EF["LF_EF"]["objective"] = extract_objective(results_dict_LF)

    # HF_EF
    out = run_ef(
        model_data_dict={"HF": model_data_time["HF"]},
        builder_dict={"HF": uHF_builder_multitime_period},
        first_stage_variables=first_stage_vars,
    )
    results_dict_HF = out["results"].to_dict()
    results_summary_EF["HF_EF"]["cpu_time"] = out["cpu_time"]
    results_summary_EF["HF_EF"]["objective"] = extract_objective(results_dict_HF)

    # MF_EF
    out = run_ef(
        model_data_dict={
            "HF": model_data_time_MF["HF"],
            "LF": model_data_time_MF["LF"],
        },
        builder_dict={
            "HF": uHF_builder_multitime_period,
            "LF": HF_builder_multitime_period,
        },
        first_stage_variables=first_stage_vars,
        bundle_config={"scheme": "mf_random", "LF": 2, "seed": 1234567890},
    )
    results_dict_MF = out["results"].to_dict()
    results_summary_EF["MF_EF"]["cpu_time"] = out["cpu_time"]
    results_summary_EF["MF_EF"]["objective"] = extract_objective(results_dict_MF)

    print_summary("Summary of EF Results", results_summary_EF)

if __name__ == "__main__":
    main()
