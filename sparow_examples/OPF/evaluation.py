from sparow.sp import stochastic_program

def extract_solution_dict(results_dict):
    key = next(iter(results_dict["solutions"]))
    return results_dict["solutions"][key]

def extract_objective(results_dict):
    solution = extract_solution_dict(results_dict)
    return solution["objectives"][0]["value"]

def extract_variable_map(results_dict):
    solution = extract_solution_dict(results_dict)
    return {var["name"]: var["value"] for var in solution["variables"]}

def compute_percent_error(true_values, comparison_values, zero_tol=1e-10):
    percent_errors = {}
    skipped = []

    for name, true_value in true_values.items():
        if name in comparison_values:
            comparison_value = comparison_values[name]

            if abs(true_value) <= zero_tol:
                skipped.append(name)
                continue

            percent_error = abs(true_value - comparison_value) / abs(true_value) * 100
            percent_errors[name] = percent_error

    return percent_errors, skipped

def resolve_on_hf(model_data, builder, first_stage_solution, first_stage_variables):
    sp = stochastic_program(first_stage_variables=first_stage_variables)
    sp.initialize_model(
        name="HF",
        model_data=model_data,
        model_builder=builder,
    )

    sp.initialize_bundles(scheme="single_bundle")
    assert len(sp.bundles) == 1, f"Expected one bundle, got {len(sp.bundles)}"

    b = next(iter(sp.bundles))
    M = sp.create_subproblem(b)

    for i in M.s.index_set():
        for j, d_j in enumerate(first_stage_solution["variables"]):
            M.s[i].time_periods[1].m.pg[str(j + 1)].fix(d_j["value"])

    results = sp.solve(M, solver="ipopt", tee=True)
    return results["obj_value"]
