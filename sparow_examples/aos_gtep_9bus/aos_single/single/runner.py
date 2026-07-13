from gtep.gtep_model import ExpansionPlanningModel
from gtep.gtep_data import ExpansionPlanningData
from gtep.gtep_solution import ExpansionPlanningSolution
from pyomo.core import TransformationFactory
from pyomo.contrib.appsi.solvers.gurobi import Gurobi
from IPython import embed
import pyomo.environ as pyo
import time
import matplotlib.pyplot as plt
import numpy as np
import plotly

start_time = time.time()

data_path = "./"  # 9 BUS
data_object = ExpansionPlanningData()
data_object.load_prescient(data_path)

mod_object = ExpansionPlanningModel(
    stages=3,  # INVESTMENT PERIODS (3,4,5)
    data=data_object,
    num_reps=12,  # NUM REPRESENTATIVE DAYS (2,4,8,12)
    len_reps=24,  # 24
    num_commit=24,  # 24
    num_dispatch=1,
)

mod_object.config["alpha_scaler"] = 1.0

mod_object.config["include_commitment"] = True

mod_object.config["flow_model"] = "CP"  # COMMENT OUT FOR DC / IN FOR CP
mod_object.config["storage"] = True
mod_object.config["transmission"] = True  # TRANSMISSION INVESTMENT FLAG
mod_object.config["thermal_generation"] = True  # THERMAL GENERATION INVESTMENT FLAG
mod_object.config["renewable_generation"] = True  # RENEWABLE GENERATION INVESTMENT FLAG
mod_object.config["scale_loads"] = False  # LEAVE AS FALSE
mod_object.config["scale_texas_loads"] = False  # LEAVE AS FALSE


mod_object.create_model()


TransformationFactory("gdp.bound_pretransformation").apply_to(mod_object.model)
TransformationFactory("gdp.bigm").apply_to(mod_object.model)

opt = Gurobi()

mod_object.results = opt.solve(mod_object.model)
sol_object = ExpansionPlanningSolution()
sol_object.load_from_model(mod_object)
sol_object.dump_json("./gtep_solution.json")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")


def plot_battery_investments():
    ins = []
    oper = []
    retir = []
    dis = []
    ext = []

    for g in mod_object.model.investmentStage[1].storInstalled.keys():
        ins.append(
            pyo.value(
                mod_object.model.investmentStage[1]
                .storInstalled[g]
                .binary_indicator_var
            )
        )
        oper.append(
            pyo.value(
                mod_object.model.investmentStage[1]
                .storOperational[g]
                .binary_indicator_var
            )
        )
        retir.append(
            pyo.value(
                mod_object.model.investmentStage[1].storRetired[g].binary_indicator_var
            )
        )
        dis.append(
            pyo.value(
                mod_object.model.investmentStage[1].storDisabled[g].binary_indicator_var
            )
        )
        ext.append(
            pyo.value(
                mod_object.model.investmentStage[1].storExtended[g].binary_indicator_var
            )
        )
    print(ins)
    print(oper)
    print(retir)
    print(dis)

    ins_2 = []
    oper_2 = []
    retir_2 = []
    dis_2 = []
    ext_2 = []
    for g in mod_object.model.investmentStage[2].storInstalled.keys():
        ins_2.append(
            pyo.value(
                mod_object.model.investmentStage[2]
                .storInstalled[g]
                .binary_indicator_var
            )
        )
        oper_2.append(
            pyo.value(
                mod_object.model.investmentStage[2]
                .storOperational[g]
                .binary_indicator_var
            )
        )
        retir_2.append(
            pyo.value(
                mod_object.model.investmentStage[2].storRetired[g].binary_indicator_var
            )
        )
        dis_2.append(
            pyo.value(
                mod_object.model.investmentStage[2].storDisabled[g].binary_indicator_var
            )
        )
        ext_2.append(
            pyo.value(
                mod_object.model.investmentStage[2].storExtended[g].binary_indicator_var
            )
        )
    print(ins_2)
    print(oper_2)
    print(retir_2)
    print(dis_2)

    ins_3 = []
    oper_3 = []
    retir_3 = []
    dis_3 = []
    ext_3 = []
    for g in mod_object.model.investmentStage[3].storInstalled.keys():
        ins_3.append(
            pyo.value(
                mod_object.model.investmentStage[3]
                .storInstalled[g]
                .binary_indicator_var
            )
        )
        oper_3.append(
            pyo.value(
                mod_object.model.investmentStage[3]
                .storOperational[g]
                .binary_indicator_var
            )
        )
        retir_3.append(
            pyo.value(
                mod_object.model.investmentStage[3].storRetired[g].binary_indicator_var
            )
        )
        dis_3.append(
            pyo.value(
                mod_object.model.investmentStage[3].storDisabled[g].binary_indicator_var
            )
        )
        ext_3.append(
            pyo.value(
                mod_object.model.investmentStage[3].storExtended[g].binary_indicator_var
            )
        )
    print(ins_3)
    print(oper_3)
    print(retir_3)
    print(dis_3)

    # sol_object.plot_levels(save_dir="./plots/")

    # # Set up bar width and positions
    generators_stage_1 = list(mod_object.model.investmentStage[1].storInstalled.keys())
    generators_stage_2 = list(mod_object.model.investmentStage[2].storInstalled.keys())
    generators_stage_3 = list(mod_object.model.investmentStage[3].storInstalled.keys())
    bar_width = 0.2
    x_indices_stage_1 = np.arange(len(generators_stage_1))
    x_indices_stage_2 = np.arange(len(generators_stage_2))
    x_indices_stage_3 = np.arange(len(generators_stage_3))

    # # Create the first bar plot for investment stage 1
    plt.figure(figsize=(12, 6))
    plt.bar(
        x_indices_stage_1,
        ins,
        width=bar_width,
        label="Installed",
        color="b",
        align="center",
    )
    plt.bar(
        x_indices_stage_1 + bar_width,
        oper,
        width=bar_width,
        label="Operational",
        color="g",
        align="center",
    )
    plt.bar(
        x_indices_stage_1 + 2 * bar_width,
        retir,
        width=bar_width,
        label="Retired",
        color="r",
        align="center",
    )
    plt.bar(
        x_indices_stage_1 + 3 * bar_width,
        ext,
        width=bar_width,
        label="Extended",
        color="y",
        align="center",
    )

    plt.xlabel("Batteries")
    plt.ylabel("Binary Indicator Values")
    plt.title("Investment Stage 1")
    plt.xticks(x_indices_stage_1 + bar_width, generators_stage_1, rotation=45)
    plt.yticks([0, 1], ["0", "1"])
    plt.legend()
    plt.tight_layout()
    plt.savefig("Bat_Invest_1.png")

    # # Create the second bar plot for investment stage 2
    plt.figure(figsize=(12, 6))
    plt.bar(
        x_indices_stage_2,
        ins_2,
        width=bar_width,
        label="Installed",
        color="b",
        align="center",
    )
    plt.bar(
        x_indices_stage_2 + bar_width,
        oper_2,
        width=bar_width,
        label="Operational",
        color="g",
        align="center",
    )
    plt.bar(
        x_indices_stage_2 + 2 * bar_width,
        retir_2,
        width=bar_width,
        label="Retired",
        color="r",
        align="center",
    )
    plt.bar(
        x_indices_stage_2 + 3 * bar_width,
        ext_2,
        width=bar_width,
        label="Extended",
        color="y",
        align="center",
    )

    plt.xlabel("Batteries")
    plt.ylabel("Binary Indicator Values")
    plt.title("Investment Stage 2")
    plt.xticks(x_indices_stage_2 + bar_width, generators_stage_2, rotation=45)
    plt.yticks([0, 1], ["0", "1"])
    plt.legend()
    plt.tight_layout()
    plt.savefig("Bat_Invest_2.png")
    # # Create the second bar plot for investment stage 3
    plt.figure(figsize=(12, 6))
    plt.bar(
        x_indices_stage_3,
        ins_3,
        width=bar_width,
        label="Installed",
        color="b",
        align="center",
    )
    plt.bar(
        x_indices_stage_3 + bar_width,
        oper_3,
        width=bar_width,
        label="Operational",
        color="g",
        align="center",
    )
    plt.bar(
        x_indices_stage_3 + 2 * bar_width,
        retir_3,
        width=bar_width,
        label="Retired",
        color="r",
        align="center",
    )
    plt.bar(
        x_indices_stage_3 + 3 * bar_width,
        ext_3,
        width=bar_width,
        label="Extended",
        color="y",
        align="center",
    )

    plt.xlabel("Batteries")
    plt.ylabel("Binary Indicator Values")
    plt.title("Investment Stage 3")
    plt.xticks(x_indices_stage_3 + bar_width, generators_stage_3, rotation=45)
    plt.yticks([0, 1], ["0", "1"])
    plt.legend()
    plt.tight_layout()
    plt.savefig("Bat_Invest_3.png")
    return


plot_battery_investments()


def sum_all_load_shed_new():
    total = 0
    cost_dispatch_total = 0
    invest_stages = mod_object.model.investmentStage.keys()

    for IS in invest_stages:
        rep_stages = mod_object.model.investmentStage[IS].representativePeriod.keys()

        for rp in rep_stages:
            com_stages = (
                mod_object.model.investmentStage[IS]
                .representativePeriod[rp]
                .commitmentPeriod.keys()
            )

            for cp in com_stages:
                dispatch_stages = (
                    mod_object.model.investmentStage[IS]
                    .representativePeriod[rp]
                    .commitmentPeriod[cp]
                    .dispatchPeriod.keys()
                )

                for dp in dispatch_stages:
                    cost_dispatch = pyo.value(
                        mod_object.model.investmentStage[IS]
                        .representativePeriod[rp]
                        .commitmentPeriod[cp]
                        .dispatchPeriod[dp]
                        .operatingCostDispatch
                    )
                    print(
                        f"Investment Stage: {IS}, Representative Period: {rp}, Commitment Period: {cp}, Dispatch Period: {dp}, Cost Dispatch: {cost_dispatch}"
                    )
                    load_shed_sum = pyo.value(
                        pyo.summation(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .loadShed
                        )
                    )
                    print(
                        f"Investment Stage: {IS}, Representative Period: {rp}, Commitment Period: {cp}, Dispatch Period: {dp}, Load Shed Sum: {load_shed_sum}"
                    )
                    total = total + load_shed_sum
                    cost_dispatch_total += cost_dispatch
    return total


total = sum_all_load_shed_new()
print("Total Load Shed Value:")
print(total)


def plot_load_shed_generation_bar_plot():
    storage_charged = []
    storage_discharged = []
    load_shed = []
    thermal_generation = []
    renewable_generation = []
    invest_stages = mod_object.model.investmentStage.keys()

    for IS in invest_stages:
        rep_stages = mod_object.model.investmentStage[IS].representativePeriod.keys()

        for rp in rep_stages:
            com_stages = (
                mod_object.model.investmentStage[IS]
                .representativePeriod[rp]
                .commitmentPeriod.keys()
            )

            for cp in com_stages:
                dispatch_stages = (
                    mod_object.model.investmentStage[IS]
                    .representativePeriod[rp]
                    .commitmentPeriod[cp]
                    .dispatchPeriod.keys()
                )

                for dp in dispatch_stages:
                    load_shed_sum = pyo.value(
                        pyo.summation(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .loadShed
                        )
                    )
                    thermal_gen_sum = pyo.value(
                        pyo.summation(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration
                        )
                    )
                    renew_sum = pyo.value(
                        pyo.summation(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .renewableGeneration
                        )
                    )
                    stor_char = pyo.value(
                        pyo.summation(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .storageCharged
                        )
                    )
                    stor_dis = pyo.value(
                        pyo.summation(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .storageDischarged
                        )
                    )
                    storage_charged.append(stor_char)
                    storage_discharged.append(stor_dis)
                    load_shed.append(load_shed_sum)
                    thermal_generation.append(thermal_gen_sum)
                    renewable_generation.append(renew_sum)
    # Step 3: Create a bar plot

    storage_charged = storage_charged[-35:]
    storage_discharged = storage_discharged[-35:]
    load_shed = load_shed[-35:]
    thermal_generation = thermal_generation[-35:]
    renewable_generation = renewable_generation[-35:]
    bar_width = 0.15  # Width of the bars
    index = np.arange(len(load_shed))  # X locations for the groups
    plt.figure(figsize=(15, 8))
    # Create the bar plots
    plt.bar(index, storage_charged, bar_width, label="Storage Charged", color="b")
    plt.bar(
        index + bar_width,
        storage_discharged,
        bar_width,
        label="Storage Discharged",
        color="g",
    )
    plt.bar(index + 2 * bar_width, load_shed, bar_width, label="Load Shed", color="r")
    plt.bar(
        index + 3 * bar_width,
        thermal_generation,
        bar_width,
        label="Thermal Generation",
        color="c",
    )
    plt.bar(
        index + 4 * bar_width,
        renewable_generation,
        bar_width,
        label="Renewable Generation",
        color="m",
    )

    # Step 4: Customize the plot
    plt.xlabel("Dispatch Periods")
    plt.ylabel("Real Power (MW)")
    # plt.title('Load Shed and Generation Overview')
    plt.xticks(
        index + 2 * bar_width, [f"{i+1}" for i in range(len(load_shed))]
    )  # Set x-tick labels
    plt.legend()
    plt.tight_layout()  # Adjust layout to prevent clipping of labels

    # Step 5: Show the plot
    plt.savefig("load_shed.png")
    return


plot_load_shed_generation_bar_plot()


def plot_objective_values():
    OC_I1 = pyo.value(mod_object.model.investmentStage[1].operatingCostInvestment)
    OC_I2 = pyo.value(mod_object.model.investmentStage[2].operatingCostInvestment)
    OC_I3 = pyo.value(mod_object.model.investmentStage[3].operatingCostInvestment)

    ic_I1 = pyo.value(mod_object.model.investmentStage[1].investment_cost)
    ic_I2 = pyo.value(mod_object.model.investmentStage[2].investment_cost)
    ic_I3 = pyo.value(mod_object.model.investmentStage[3].investment_cost)
    # Prepare data for plotting
    stages = ["Investment Stage 1", "Investment Stage 2", "Investment Stage 3"]
    operating_costs = [OC_I1, OC_I2, OC_I3]
    investment_costs = [ic_I1, ic_I2, ic_I3]
    print("Operating costs:")
    print(operating_costs)
    print("Investment costs:")
    print(investment_costs)
    # Set bar width and positions
    plt.figure(figsize=(15, 8))
    bar_width = 0.35
    index = np.arange(len(stages))

    # Create the bar plot
    plt.bar(
        index,
        [x / 1e6 for x in operating_costs],
        bar_width,
        label="Operating Costs",
        color="b",
    )
    plt.bar(
        index + bar_width,
        [x / 1e6 for x in investment_costs],
        bar_width,
        label="Investment Costs",
        color="g",
    )

    # Customize the plot
    # plt.xlabel('Investment Stages')
    plt.ylabel("Cost (Million $)")
    # plt.title('Operating and Investment Costs by Investment Stage')
    plt.xticks(index + bar_width / 2, stages)  # Set x-tick labels
    plt.legend()
    plt.tight_layout()  # Adjust layout to prevent clipping of labels

    # Show the plot
    plt.savefig("investment.png")
    return


plot_objective_values()


def plot_load_profiles():
    # Step 1: Retrieve load values for each bus
    bus_loads = {}
    for bus in mod_object.model.data.md.data["elements"]["load"].keys():
        bus_loads[bus] = mod_object.model.data.md.data["elements"]["load"][bus][
            "p_load"
        ]["values"]

    # Step 2: Sum the load values as vectors
    # Assuming all load lists are of the same length
    total_load = np.sum([np.array(bus_loads[bus]) for bus in bus_loads], axis=0)

    # Step 3: Create a bar plot of the total load over time
    time_points = np.arange(
        len(total_load)
    )  # Assuming each entry corresponds to a time point
    plt.figure(figsize=(15, 8), facecolor="white")

    plt.bar(time_points, total_load, edgecolor="g")  # Adjust width as needed

    plt.xticks(
        ticks=np.arange(0, 8640, 24 * 30),  # Every 30 days
        labels=[
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
    )
    plt.xlabel("Months")
    plt.ylabel("Total Load (MW)")
    # plt.title('Total Load Over Time for Three Buses')
    plt.tight_layout()  # Adjust layout to prevent clipping of labels

    # Step 4: Show the plot
    plt.savefig("Load_Demand.png")

    return


plot_load_profiles()


def plot_objective_cats_by_dispatch():
    cost_dispatch = []
    generation_cost_dispatch = []
    load_shed_cost = []
    batterycost_dispatch = []

    invest_stages = mod_object.model.investmentStage.keys()

    for IS in invest_stages:
        rep_stages = mod_object.model.investmentStage[IS].representativePeriod.keys()

        for rp in rep_stages:
            com_stages = (
                mod_object.model.investmentStage[IS]
                .representativePeriod[rp]
                .commitmentPeriod.keys()
            )

            for cp in com_stages:
                dispatch_stages = (
                    mod_object.model.investmentStage[IS]
                    .representativePeriod[rp]
                    .commitmentPeriod[cp]
                    .dispatchPeriod.keys()
                )

                for dp in dispatch_stages:
                    cost_dispatch.append(
                        pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .operatingCostDispatch
                        )
                    )
                    generation_cost_dispatch.append(
                        pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .generationCostDispatch
                        )
                    )
                    load_shed_cost.append(
                        pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .loadShedCostDispatch
                        )
                    )
                    batterycost_dispatch.append(
                        pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .storageCostDispatch
                        )
                    )

    # Step 3: Create a bar plot
    cost_dispatch = cost_dispatch[:20]
    generation_cost_dispatch = generation_cost_dispatch[:20]
    load_shed_cost = load_shed_cost[:20]
    batterycost_dispatch = batterycost_dispatch[:20]

    bar_width = 0.15  # Width of the bars
    index = np.arange(len(cost_dispatch))  # X locations for the groups
    plt.figure(figsize=(15, 8))
    # Create the bar plots
    plt.bar(index, cost_dispatch, bar_width, label="Dispatch Cost", color="b")
    plt.bar(
        index + bar_width,
        generation_cost_dispatch,
        bar_width,
        label="Generation Cost",
        color="g",
    )
    plt.bar(
        index + 2 * bar_width,
        load_shed_cost,
        bar_width,
        label="Load Shed Cost",
        color="r",
    )
    plt.bar(
        index + 3 * bar_width,
        batterycost_dispatch,
        bar_width,
        label="Battery Cost",
        color="c",
    )

    # Step 4: Customize the plot
    plt.xlabel("Dispatch Periods")
    plt.ylabel("Values ($)")
    # plt.title('Cost Categories by Dispatch Period')
    plt.xticks(
        index + 2 * bar_width, [f"{i+1}" for i in range(len(load_shed_cost))]
    )  # Set x-tick labels
    plt.legend()
    plt.tight_layout()  # Adjust layout to prevent clipping of labels

    # Step 5: Show the plot
    plt.savefig("Costing_categories.png")
    return


plot_objective_cats_by_dispatch()


def plot_bar_broad_cats():
    storage_charged = []
    storage_discharged = []
    load_shed = []
    thermal_generation = []
    renewable_generation = []
    PV = []
    wind = []
    hydro = []
    gas = []
    coal = []
    nuclear = []
    other = []

    invest_stages = mod_object.model.investmentStage.keys()

    for IS in invest_stages:
        rep_stages = mod_object.model.investmentStage[IS].representativePeriod.keys()

        for rp in rep_stages:
            com_stages = (
                mod_object.model.investmentStage[IS]
                .representativePeriod[rp]
                .commitmentPeriod.keys()
            )

            for cp in com_stages:
                dispatch_stages = (
                    mod_object.model.investmentStage[IS]
                    .representativePeriod[rp]
                    .commitmentPeriod[cp]
                    .dispatchPeriod.keys()
                )

                for dp in dispatch_stages:
                    load_shed_sum = pyo.value(
                        pyo.summation(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .loadShed
                        )
                    )
                    thermal_gen_sum = pyo.value(
                        pyo.summation(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration
                        )
                    )
                    renew_sum = pyo.value(
                        pyo.summation(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .renewableGeneration
                        )
                    )
                    stor_char = pyo.value(
                        pyo.summation(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .storageCharged
                        )
                    )
                    stor_dis = pyo.value(
                        pyo.summation(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .storageDischarged
                        )
                    )
                    storage_charged.append(stor_char)
                    storage_discharged.append(stor_dis)
                    load_shed.append(load_shed_sum)
                    thermal_generation.append(thermal_gen_sum)
                    renewable_generation.append(renew_sum)
                    g = (
                        pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G1_R1"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G1_R2"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G1_R3"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G2_R1"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G2_R2"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G2_R3"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G8_R1"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G8_R2"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G8_R3"]
                        )
                    )
                    c = (
                        pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G4_R1"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G4_R2"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G4_R3"]
                        )
                    )
                    n = (
                        pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G7_R1"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G7_R2"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G7_R3"]
                        )
                    )
                    o = (
                        pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G10_R1"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G10_R2"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G10_R3"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G22_R1"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G22_R2"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G22_R3"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G6_R1"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G6_R2"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .thermalGeneration["G6_R3"]
                        )
                    )
                    gas.append(g)
                    coal.append(c)
                    nuclear.append(n)
                    other.append(o)
                    p = (
                        pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .renewableGeneration["Solar_R1"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .renewableGeneration["Solar_R2"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .renewableGeneration["Solar_R3"]
                        )
                    )
                    w = (
                        pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .renewableGeneration["Wind_R1"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .renewableGeneration["Wind_R2"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .renewableGeneration["Wind_R3"]
                        )
                    )
                    h = (
                        pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .renewableGeneration["Hydro_R1"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .renewableGeneration["Hydro_R2"]
                        )
                        + pyo.value(
                            mod_object.model.investmentStage[IS]
                            .representativePeriod[rp]
                            .commitmentPeriod[cp]
                            .dispatchPeriod[dp]
                            .renewableGeneration["Hydro_R3"]
                        )
                    )
                    PV.append(p)
                    wind.append(w)
                    hydro.append(h)

    # Loop through the data in chunks of 24
    for i in range(0, len(load_shed), 24):
        # Set up the bar width
        bar_width = 0.5  # Width of the bars

        # Initialize the figure
        plt.figure(figsize=(15, 8))
        # Extract the current chunk
        chunk_storage_charged = storage_charged[i : i + 24]
        chunk_storage_discharged = storage_discharged[i : i + 24]
        chunk_load_shed = load_shed[i : i + 24]
        chunk_thermal_generation = thermal_generation[i : i + 24]
        chunk_renewable_generation = renewable_generation[i : i + 24]

        # Set up the index for the current chunk
        index = np.arange(len(chunk_load_shed))  # X locations for the groups

        # Create the stacked bar plots for the current chunk
        plt.bar(
            index, chunk_storage_charged, bar_width, label="Storage Charged", color="b"
        )
        plt.bar(
            index,
            chunk_storage_discharged,
            bar_width,
            bottom=chunk_storage_charged,
            label="Storage Discharged",
            color="g",
        )
        plt.bar(
            index,
            chunk_load_shed,
            bar_width,
            bottom=np.array(chunk_storage_charged) + np.array(chunk_storage_discharged),
            label="Load Shed",
            color="r",
        )
        plt.bar(
            index,
            chunk_thermal_generation,
            bar_width,
            bottom=np.array(chunk_storage_charged)
            + np.array(chunk_storage_discharged)
            + np.array(chunk_load_shed),
            label="Thermal Generation",
            color="c",
        )
        plt.bar(
            index,
            chunk_renewable_generation,
            bar_width,
            bottom=np.array(chunk_storage_charged)
            + np.array(chunk_storage_discharged)
            + np.array(chunk_load_shed)
            + np.array(chunk_thermal_generation),
            label="Renewable Generation",
            color="m",
        )

        # Customize the plot for the current chunk
        plt.xlabel("Dispatch Periods")
        plt.ylabel("Real Power (MW)")
        plt.xticks(
            index, [f"{i+1}" for i in range(len(chunk_load_shed))]
        )  # Set x-tick labels
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(
            handles[::-1], labels[::-1], loc="center left", bbox_to_anchor=(1, 0.5)
        )
        plt.tight_layout()  # Adjust layout to prevent clipping of labels

        # Show the plot for the current chunk
        plt.savefig(str(i) + "bar_plot_broad_cat.png")
        # Loop through the data in chunks of 24
    for i in range(0, len(load_shed), 24):
        # Set up the bar width
        bar_width = 0.5  # Width of the bars

        # Initialize the figure
        plt.figure(figsize=(15, 8))
        # Extract the current chunk
        chunk_storage_charged = storage_charged[i : i + 24]
        chunk_storage_discharged = storage_discharged[i : i + 24]
        chunk_load_shed = load_shed[i : i + 24]
        chunk_thermal_generation = thermal_generation[i : i + 24]
        chunk_renewable_generation = renewable_generation[i : i + 24]
        chunk_gas = gas[i : i + 24]
        chunk_coal = coal[i : i + 24]
        chunk_nuclear = nuclear[i : i + 24]
        chunk_other = other[i : i + 24]
        chunk_PV = PV[i : i + 24]
        chunk_wind = wind[i : i + 24]
        chunk_hydro = hydro[i : i + 24]
        # Set up the index for the current chunk
        index = np.arange(len(chunk_load_shed))  # X locations for the groups

        # Create the stacked bar plots
        plt.bar(
            index, chunk_storage_charged, bar_width, label="Storage Charged", color="b"
        )
        plt.bar(
            index,
            chunk_storage_discharged,
            bar_width,
            bottom=chunk_storage_charged,
            label="Storage Discharged",
            color="g",
        )
        plt.bar(
            index,
            chunk_load_shed,
            bar_width,
            bottom=np.array(chunk_storage_charged) + np.array(chunk_storage_discharged),
            label="Load Shed",
            color="r",
        )

        # Stack the generation categories
        plt.bar(
            index,
            chunk_gas,
            bar_width,
            bottom=np.array(chunk_storage_charged)
            + np.array(chunk_storage_discharged)
            + np.array(chunk_load_shed),
            label="Gas",
            color="c",
        )
        plt.bar(
            index,
            chunk_coal,
            bar_width,
            bottom=np.array(chunk_storage_charged)
            + np.array(chunk_storage_discharged)
            + np.array(chunk_load_shed)
            + np.array(chunk_gas),
            label="Coal",
            color="brown",
        )
        plt.bar(
            index,
            chunk_nuclear,
            bar_width,
            bottom=np.array(chunk_storage_charged)
            + np.array(chunk_storage_discharged)
            + np.array(chunk_load_shed)
            + np.array(chunk_gas)
            + np.array(chunk_coal),
            label="Nuclear",
            color="purple",
        )
        plt.bar(
            index,
            chunk_other,
            bar_width,
            bottom=np.array(chunk_storage_charged)
            + np.array(chunk_storage_discharged)
            + np.array(chunk_load_shed)
            + np.array(chunk_gas)
            + np.array(chunk_coal)
            + np.array(chunk_nuclear),
            label="Other",
            color="gray",
        )
        plt.bar(
            index,
            chunk_PV,
            bar_width,
            bottom=np.array(chunk_storage_charged)
            + np.array(chunk_storage_discharged)
            + np.array(chunk_load_shed)
            + np.array(chunk_gas)
            + np.array(chunk_coal)
            + np.array(chunk_nuclear)
            + np.array(chunk_other),
            label="Solar",
            color="orange",
        )
        plt.bar(
            index,
            chunk_wind,
            bar_width,
            bottom=np.array(chunk_storage_charged)
            + np.array(chunk_storage_discharged)
            + np.array(chunk_load_shed)
            + np.array(chunk_gas)
            + np.array(chunk_coal)
            + np.array(chunk_nuclear)
            + np.array(chunk_PV)
            + np.array(chunk_other),
            label="Wind",
            color="yellow",
        )
        plt.bar(
            index,
            chunk_hydro,
            bar_width,
            bottom=np.array(chunk_storage_charged)
            + np.array(chunk_storage_discharged)
            + np.array(chunk_load_shed)
            + np.array(chunk_gas)
            + np.array(chunk_coal)
            + np.array(chunk_nuclear)
            + np.array(chunk_PV)
            + np.array(chunk_wind)
            + np.array(chunk_other),
            label="Hydro",
            color="black",
        )
        # Customize the plot for the current chunk
        plt.xlabel("Dispatch Periods")
        plt.ylabel("Real Power (MW)")
        plt.xticks(
            index, [f"{i+1}" for i in range(len(chunk_load_shed))]
        )  # Set x-tick labels
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(
            handles[::-1], labels[::-1], loc="center left", bbox_to_anchor=(1, 0.5)
        )
        plt.tight_layout()  # Adjust layout to prevent clipping of labels

        # Show the plot for the current chunk
        plt.savefig(str(i) + "bar_plot_all_cat.png")
    for i in range(0, len(load_shed), 24):
        # Set up the bar width
        bar_width = 0.5  # Width of the bars
        # Extract the current chunk
        chunk_storage_charged = storage_charged[i : i + 24]
        chunk_storage_discharged = storage_discharged[i : i + 24]
        chunk_load_shed = load_shed[i : i + 24]
        chunk_thermal_generation = thermal_generation[i : i + 24]
        chunk_renewable_generation = renewable_generation[i : i + 24]
        chunk_gas = gas[i : i + 24]
        chunk_coal = coal[i : i + 24]
        chunk_nuclear = nuclear[i : i + 24]
        chunk_other = other[i : i + 24]
        chunk_PV = PV[i : i + 24]
        chunk_wind = wind[i : i + 24]
        chunk_hydro = hydro[i : i + 24]
        # Set up the index for the current chunk
        index = np.arange(len(chunk_load_shed))  # X locations for the groups
        # Set up the x locations for the groups
        x = np.arange(len(chunk_load_shed))  # X locations for the groups

        # Initialize the figure
        plt.figure(figsize=(15, 8))

        # Create the stack plot
        plt.stackplot(
            x,
            chunk_storage_charged,
            chunk_storage_discharged,
            chunk_load_shed,
            chunk_gas,
            chunk_coal,
            chunk_nuclear,
            chunk_other,
            chunk_PV,
            chunk_wind,
            chunk_hydro,
            labels=[
                "Storage Charged",
                "Storage Discharged",
                "Load Shed",
                "Gas",
                "Coal",
                "Nuclear",
                "Other",
                "Solar",
                "Wind",
                "Hydro",
            ],
            colors=[
                "b",
                "g",
                "r",
                "c",
                "brown",
                "purple",
                "gray",
                "orange",
                "yellow",
                "black",
            ],
        )

        # Step 4: Customize the plot
        plt.xlabel("Dispatch Periods")
        plt.ylabel("Real Power (MW)")
        # plt.title('Load Shed and Generation Overview')
        plt.xticks(
            x, [f"{i+1}" for i in range(len(chunk_load_shed))]
        )  # Set x-tick labels
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(
            handles[::-1], labels[::-1], loc="center left", bbox_to_anchor=(1, 0.5)
        )
        # plt.legend(loc='upper left')
        plt.tight_layout()  # Adjust layout to prevent clipping of labels
        plt.savefig(str(i) + "stack_plot_all_cat.png")

    return


plot_bar_broad_cats()
