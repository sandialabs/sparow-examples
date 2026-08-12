from config import ExperimentConfig
from scenarios import build_model_data_time, split_multifidelity
from model_builders import (
    LF_builder_multitime_period,
    HF_builder_multitime_period,
    uHF_builder_multitime_period,
)
from experiment_runner import run_ef, run_ph
from evaluation import (
    extract_objective,
    extract_solution_dict,
    extract_variable_map,
    compute_percent_error,
    resolve_on_hf,
)
from reporting import print_summary, plot_opf_runs
from IPython import embed

def main():
    cfg = ExperimentConfig()
    first_stage_vars = ["time_periods[1].m.pg[*]"]

    # Build scenario data
    model_data_time = build_model_data_time(
        scenarios=cfg.scenarios,
        bound=cfg.bound,
        data_dir=cfg.data_dir,
        seed=cfg.seed,
    )
    model_data_time_MF = split_multifidelity(model_data_time)

    # Store raw result dicts for later HF re-evaluation
    stored_results = {}

    # -------------------------
    # EF Experiments
    # -------------------------
    ef_summary = {
        "HF_EF": {"cpu_time": None, "objective": None},
        "LF_EF": {"cpu_time": None, "objective": None},
        "MF_EF": {"cpu_time": None, "objective": None},
    }

    ef_experiments = [
        {
            "name": "LF_EF",
            "model_data_dict": {"HF": model_data_time["HF"]},
            "builder_dict": {"HF": HF_builder_multitime_period},
            "bundle_config": None,
        },
        {
            "name": "HF_EF",
            "model_data_dict": {"HF": model_data_time["HF"]},
            "builder_dict": {"HF": uHF_builder_multitime_period},
            "bundle_config": None,
        },
        {
            "name": "MF_EF",
            "model_data_dict": {
                "HF": model_data_time_MF["HF"],
                "LF": model_data_time_MF["LF"],
            },
            "builder_dict": {
                "HF": uHF_builder_multitime_period,
                "LF": HF_builder_multitime_period,
            },
            "bundle_config": {"scheme": "mf_random", "LF": 2, "seed": 1234567890},
        },
    ]

    for exp in ef_experiments:
        print("-" * 60)
        print(f"Running {exp['name']}")
        print("-" * 60)

        out = run_ef(
            model_data_dict=exp["model_data_dict"],
            builder_dict=exp["builder_dict"],
            first_stage_variables=first_stage_vars,
            bundle_config=exp["bundle_config"],
        )

        results_dict = out["results"].to_dict()
        stored_results[exp["name"]] = results_dict
        ef_summary[exp["name"]]["cpu_time"] = out["cpu_time"]
        ef_summary[exp["name"]]["objective"] = extract_objective(results_dict)


    # -------------------------
    # PH Experiments
    # -------------------------
    ph_summary = {
        "HF_PH": {"cpu_time": None, "objective": None},
        "LF_PH": {"cpu_time": None, "objective": None},
        "MF_PH_Random": {"cpu_time": None, "objective": None},
        "MF_PH_Similar": {"cpu_time": None, "objective": None},
        "MF_PH_Dissimilar": {"cpu_time": None, "objective": None},
    }

    ph_experiments = [
        {
            "name": "LF_PH",
            "model_data_dict": {"HF": model_data_time["HF"]},
            "builder_dict": {"HF": HF_builder_multitime_period},
            "bundle_config": None,
        },
        {
            "name": "HF_PH",
            "model_data_dict": {"HF": model_data_time["HF"]},
            "builder_dict": {"HF": uHF_builder_multitime_period},
            "bundle_config": None,
        },
        {
            "name": "MF_PH_Random",
            "model_data_dict": {
                "HF": model_data_time_MF["HF"],
                "LF": model_data_time_MF["LF"],
            },
            "builder_dict": {
                "HF": uHF_builder_multitime_period,
                "LF": HF_builder_multitime_period,
            },
            "bundle_config": {"scheme": "mf_random", "LF": 2, "seed": 1234567890},
        },
        {
            "name": "MF_PH_Similar",
            "model_data_dict": {
                "HF": model_data_time_MF["HF"],
                "LF": model_data_time_MF["LF"],
            },
            "builder_dict": {
                "HF": uHF_builder_multitime_period,
                "LF": HF_builder_multitime_period,
            },
            "bundle_config": {
                "scheme": "mf_kmeans_similar",
                "LF": 2,
                "seed": 1234567890,
                "data_key": "DEMAND_2",
            },
        },
        {
            "name": "MF_PH_Dissimilar",
            "model_data_dict": {
                "HF": model_data_time_MF["HF"],
                "LF": model_data_time_MF["LF"],
            },
            "builder_dict": {
                "HF": uHF_builder_multitime_period,
                "LF": HF_builder_multitime_period,
            },
            "bundle_config": {
                "scheme": "mf_kmeans_dissimilar",
                "LF": 2,
                "seed": 1234567890,
                "data_key": "DEMAND_2",
            },
        },
    ]

    for exp in ph_experiments:
        print("-" * 60)
        print(f"Running {exp['name']}")
        print("-" * 60)

        out = run_ph(
            model_data_dict=exp["model_data_dict"],
            builder_dict=exp["builder_dict"],
            first_stage_variables=first_stage_vars,
            max_iterations=cfg.max_iterations,
            default_rho=cfg.dr,
            bundle_config=exp["bundle_config"],
        )

        results_dict = out["results"].to_dict()
        stored_results[exp["name"]] = results_dict
        ph_summary[exp["name"]]["cpu_time"] = out["cpu_time"]
        ph_summary[exp["name"]]["objective"] = extract_objective(results_dict)

    # -------------------------
    # Optional percent error comparison
    # -------------------------
    hf_vars = extract_variable_map(stored_results["HF_EF"])


    for name in ["LF_EF", "MF_EF", "HF_PH", "LF_PH", "MF_PH_Random", "MF_PH_Similar", "MF_PH_Dissimilar"]:
        if name in stored_results:
            comp_vars = extract_variable_map(stored_results[name])
            errors, skipped = compute_percent_error(hf_vars, comp_vars)
            avg_err = sum(errors.values()) / len(errors) if errors else float("nan")
            print(f"Average percent error for {name} vs HF_EF: {avg_err:.4f}%")
            print(f"Skipped {len(skipped)} zero-reference variables")


    # -------------------------
    # Re-evaluate all methods on HF
    # -------------------------
    reevaluated_summary = {}

    methods_to_reevaluate = [
        "HF_EF",
        "LF_EF",
        "MF_EF",
        "HF_PH",
        "LF_PH",
        "MF_PH_Random",
        "MF_PH_Similar",
        "MF_PH_Dissimilar",
    ]

    for method in methods_to_reevaluate:
        print("-" * 60)
        print(f"Re-evaluating {method} on HF model")
        print("-" * 60)

        solution = extract_solution_dict(stored_results[method])

        hf_obj = resolve_on_hf(
            model_data=model_data_time["HF"],
            builder=uHF_builder_multitime_period,
            first_stage_solution=solution,
            first_stage_variables=first_stage_vars,
        )

        reevaluated_summary[method] = {"objective": hf_obj}
    
    print_summary("Summary of EF Results", ef_summary)
    print_summary("Summary of PH Results", ph_summary)

    print("\nSummary of HF Re-evaluated Objectives")
    print("-" * 60)
    for method, result in reevaluated_summary.items():
        print(f"{method}:")
        print(f"  HF Objective Value: {result['objective']:.4f}")
    print("-" * 60)
    
    plot_opf_runs(
        ef_summary=ef_summary,
        ph_summary=ph_summary,
        reevaluated_summary=reevaluated_summary,
        filename="OPF_runs.png",
    )   


if __name__ == "__main__":
    main()
