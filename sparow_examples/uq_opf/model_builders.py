from pyomo.environ import ConcreteModel, Block, Objective, ConstraintList, minimize
from egret.parsers.matpower_parser import create_ModelData
from egret.models.copperplate_dispatch import create_copperplate_dispatch_approx_model
from egret.models.dcopf import create_btheta_dcopf_model
from egret.models.acopf import create_psv_acopf_model


def _build_multitime_model(
    data,
    single_period_builder,
    load_mismatch_cost,
    q_load_mismatch_cost=None,
    add_ramping=False,
):
    test_case = data["data_dir"]
    fm = ConcreteModel()
    fm.time_periods = Block([1, 2, 3, 4])
    objectives = []

    generator_names = None

    for time in fm.time_periods:
        model_data = create_ModelData(test_case)
        model_data.data["system"]["load_mismatch_cost"] = load_mismatch_cost
        if q_load_mismatch_cost is not None:
            model_data.data["system"]["q_load_mismatch_cost"] = q_load_mismatch_cost

        for load_name, load_info in model_data.data["elements"]["load"].items():
            if load_info["in_service"]:
                load_info["p_load"] *= data[f"DEMAND_{time}"]

        fm.time_periods[time].m, md = single_period_builder(
            model_data, include_feasibility_slack=True
        )

        if hasattr(fm.time_periods[time].m, "obj"):
            fm.time_periods[time].m.obj.deactivate()
            objectives.append(fm.time_periods[time].m.obj.expr)

        if generator_names is None:
            generator_names = list(model_data.data["elements"]["generator"].keys())

    fm.obj = Objective(expr=sum(objectives), sense=minimize)

    if add_ramping:
        fm.ramping = ConstraintList()
        for time in range(2, 5):
            for g_name in generator_names:
                pg_prev = fm.time_periods[time - 1].m.pg[g_name]
                pg_curr = fm.time_periods[time].m.pg[g_name]
                fm.ramping.add(expr=pg_curr <= 1.10 * pg_prev)
                fm.ramping.add(expr=pg_curr >= 0.90 * pg_prev)

    return fm


def LF_builder_multitime_period(data, args):
    return _build_multitime_model(
        data=data,
        single_period_builder=create_copperplate_dispatch_approx_model,
        load_mismatch_cost=500,
        add_ramping=False,
    )


def HF_builder_multitime_period(data, args):
    return _build_multitime_model(
        data=data,
        single_period_builder=create_btheta_dcopf_model,
        load_mismatch_cost=500,
        add_ramping=True,
    )


def uHF_builder_multitime_period(data, args):
    return _build_multitime_model(
        data=data,
        single_period_builder=create_psv_acopf_model,
        load_mismatch_cost=500,
        q_load_mismatch_cost=0,
        add_ramping=True,
    )
