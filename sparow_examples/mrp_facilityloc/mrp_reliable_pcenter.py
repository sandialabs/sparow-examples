import pyomo.environ as pyo
import itertools
import numpy as np
from sparow.sp import stochastic_program
from sparow.ci import CIProblemAdapter

import argparse
import json
import os

"""
RELIABLE P-CENTER EXAMPLE FOR ACV-MRP

Problem formulations adapted from: https://www.sciencedirect.com/science/article/pii/S0307904X19304263

All three formulations use the same population scenarios.
Interpretation:

    Stage 1 (before uncertainty): x, y
    Stage 2 (after uncertainty):  w

The sampled outer scenario represents the realized post-disruption state:
    - disrupted demand vector
    - disrupted assignment costs
    - facility availability

Models:
    1. LF classic:
       A simplified p-center model using scenario-specific realized costs/demands,
       but no second-stage recourse (surrogate model).

    2. LF stochastic:
       Two-stage stochastic reliable p-center.

    3. HF robust-like:
       Two-stage reliable p-center with a more conservative objective based on
       the realized disruption structure.
"""

# Large multiplicative constant for objective
# This helps us ensure that the optimality gap estimates are on
# the order of magnitude we'd like for downstream analysis
MULT_CONSTANT = 1.0

app_data = {}

app_data["num_demand_points"] = 4  # |I| is number of clients
app_data["num_sites"] = 4  # |J| is number of potential facility locations
app_data["p"] = 1  # number of facilities that must be opened

app_data["I"] = [f"i{i}" for i in range(app_data["num_demand_points"])]
app_data["J"] = [f"j{j}" for j in range(app_data["num_sites"])]

app_data["alpha1"] = 0.2  # weight placed on maximum transport cost in 1st stage
app_data["alpha2"] = 0.8  # weight placed on maximum transport cost in 2nd stage

# Deterministic first-stage nominal transport costs used before disruption
app_data["c_stage1"] = {
    "i0": {"j0": 1.0, "j1": 8.0, "j2": 11.0, "j3": 14.0},
    "i1": {"j0": 1.2, "j1": 8.0, "j2": 11.0, "j3": 14.0},
    "i2": {"j0": 9.5, "j1": 14.0, "j2": 1.0, "j3": 8.0},
    "i3": {"j0": 9.8, "j1": 14.0, "j2": 1.2, "j3": 8.0},
}

# Deterministic first-stage demand weights used before disruption
app_data["d_stage1"] = {
    "i0": 55.0,
    "i1": 50.0,
    "i2": 35.0,
    "i3": 30.0,
}

# ==== SCENARIO DATA ===========================================================


class ReliablePCenterScenarioData(object):
    """
    Construct a finite population of post-disruption scenarios to sample from.

    Each scenario (xi) contains:
        - realized second-stage demand vector d2(i; xi)
        - realized second-stage transport costs c2(i,j; xi)
        - realized facility availability indicator a(j; xi), where a=1 means failed

    The finite population size is controlled by num_data_points.
    """

    def __init__(self, num_data_points):
        self.num_data_points = num_data_points

        # Demand multipliers for each customer after disruption
        self.low_demand_multiplier = {
            "i0": 0.6,
            "i1": 0.6,
            "i2": 0.3,
            "i3": 0.3,
        }
        self.high_demand_multiplier = {
            "i0": 1.2,
            "i1": 1.2,
            "i2": 2.4,
            "i3": 2.4,
        }

        self.demand_multiplier_supports = {}
        for i in app_data["I"]:
            self.demand_multiplier_supports[i] = np.linspace(
                self.low_demand_multiplier[i],
                self.high_demand_multiplier[i],
                num_data_points,
            )

        # Small finite set of facility-failure patterns
        self.failure_patterns = [
            {"j0": 0, "j1": 0, "j2": 0, "j3": 0},  # no failure
            {"j0": 0, "j1": 1, "j2": 0, "j3": 0},  # j1 fails
            {"j0": 0, "j1": 0, "j2": 1, "j3": 0},  # j2 fails
        ]

    def scenario_generator(self):
        """
        Final output is a dictionary with a single key-value pair.
        The key is "scenarios"
        The value is a list, called scen_dict_list. It contains one dictionary per possible
        population scenario. Each scenario's dictionary must contain "ID" and "Probability" (along with other relevant data).
        """
        I = app_data["I"]
        J = app_data["J"]
        c_stage1 = app_data["c_stage1"]
        d_stage1 = app_data["d_stage1"]

        demand_lists = [self.demand_multiplier_supports[i] for i in I]

        # Each client has num_data_points possible demand-multiplier values, so there are num_data_points^|I| joint demand realizations,
        # and each such realization can occur with any failure pattern
        total_scens = (self.num_data_points ** len(I)) * len(self.failure_patterns)
        scen_prob = 1.0 / total_scens

        scen_id = 0  # naming convention: each scenario ID string ends in a number (population index)
        scen_dict_list = []

        # We build the full finite scenario population by combining:
        #   (i) one possible disrupted-demand multiplier choice for each client, and
        #   (ii) one possible facility-failure pattern.
        # Use itertools.product to get cartesian product
        for demand_multiplier_tuple in itertools.product(*demand_lists):

            # Each client i has a realized post-disruption
            # demand multiplier stored in multipliers[i] for this scenario.
            multipliers = {
                i: float(demand_multiplier_tuple[idx]) for idx, i in enumerate(I)
            }

            # For each realized disrupted-demand vector, pair it with every possible
            # facility-failure pattern
            for failure_pattern in self.failure_patterns:

                # The realized stage-2 demand at client i is modeled as the
                # deterministic stage-1 demand scaled by the scenario-specific multiplier.
                demand2 = {i: float(d_stage1[i] * multipliers[i]) for i in I}

                # We increase the nominal stage-1 transport cost to reflect disruption:
                # a penalty is applied if site j fails in this scenario
                cost2 = {}
                for i in I:
                    cost2[i] = {}
                    for j in J:
                        failure_penalty = 3.0 if failure_pattern[j] == 1 else 1.00
                        cost2[i][j] = float(c_stage1[i][j] * failure_penalty)

                scen_dict_list.append(
                    {
                        "ID": f"scen_{scen_id}",
                        "Demand2": demand2,
                        "Cost2": cost2,
                        "Availability": failure_pattern,  # 1 means the facility fails in this scenario
                        "Probability": scen_prob,
                    }
                )
                scen_id += 1

        print(
            f"Total number of population scenarios generated and returned: {len(scen_dict_list)}"
        )
        return {"scenarios": scen_dict_list}


HFScenarioObject = ReliablePCenterScenarioData(num_data_points=5)
ClassicLFScenarioObject = ReliablePCenterScenarioData(num_data_points=5)
StochasticLFScenarioObject = ReliablePCenterScenarioData(num_data_points=5)

HF_scendata = HFScenarioObject.scenario_generator()
ClassicLF_scendata = ClassicLFScenarioObject.scenario_generator()
StochasticLF_scendata = StochasticLFScenarioObject.scenario_generator()

# ==== MODEL DATA ===============================================================

# This is a multi-model container:
# stores scenario datasets for each model
scenario_data_by_model = {
    "HF": HF_scendata,
    "LF_classic": ClassicLF_scendata,
    "LF_stochastic": StochasticLF_scendata,
}

# ==== MODEL BUILDERS ===========================================================


def classic_pcenter_builder(data, args):
    """
    Low-fidelity classic p-center:
    uses x and y only, no recourse w.
    Uses the realized scenario's post-disruption costs/demands directly
    as a simplified surrogate model.
    So this surrogate solves a simpler deterministic problem using the realized scenario data as parameters.
    """
    I = app_data["I"]
    J = app_data["J"]
    p = app_data["p"]

    c2 = data["Cost2"]
    d2 = data["Demand2"]

    model = pyo.ConcreteModel(data["ID"])

    model.I = pyo.Set(initialize=I)
    model.J = pyo.Set(initialize=J)

    model.L = pyo.Var(within=pyo.NonNegativeReals)
    model.x = pyo.Var(model.I, model.J, within=pyo.Binary)
    model.y = pyo.Var(model.J, within=pyo.Binary)

    # Tracking maximum transportation cost
    def radius_rule(model, i):
        return model.L >= sum(c2[i][j] * d2[i] * model.x[i, j] for j in J)

    model.Radius = pyo.Constraint(model.I, rule=radius_rule)

    # Must open p facilities
    def open_p_rule(model):
        return sum(model.y[j] for j in J) == p

    model.OpenP = pyo.Constraint(rule=open_p_rule)

    # Can only assign a client to a facility if we've opened that facility
    def assign_open_rule(model, i, j):
        return model.x[i, j] <= model.y[j]

    model.AssignOpen = pyo.Constraint(model.I, model.J, rule=assign_open_rule)

    # Every client assigned to exactly one facility
    def assign_one_rule(model, i):
        return sum(model.x[i, j] for j in J) == 1

    model.AssignOne = pyo.Constraint(model.I, rule=assign_one_rule)

    # Minimize the maximum transportation cost
    # Large multiplicative constant helps ensure that the optimality gap estimates are on
    # the order of magnitude we'd like for downstream analysis
    model.obj = pyo.Objective(expr=MULT_CONSTANT * (model.L), sense=pyo.minimize)

    return model


def stochastic_reliable_pcenter_builder(data, args):
    """
    Low-fidelity stochastic reliable p-center:
    stage 1: x, y
    stage 2: w
    objective: alpha1 * L1 + alpha2 * L2

    This builder represents a scenario-wise recourse model.
    """
    I = app_data["I"]
    J = app_data["J"]
    p = app_data["p"]
    alpha1 = app_data["alpha1"]
    alpha2 = app_data["alpha2"]

    c_stage1 = app_data["c_stage1"]
    d_stage1 = app_data["d_stage1"]

    c2 = data["Cost2"]
    d2 = data["Demand2"]
    a = data["Availability"]

    model = pyo.ConcreteModel(data["ID"])

    model.I = pyo.Set(initialize=I)
    model.J = pyo.Set(initialize=J)

    model.L1 = pyo.Var(within=pyo.NonNegativeReals)
    model.L2 = pyo.Var(within=pyo.NonNegativeReals)

    model.x = pyo.Var(model.I, model.J, within=pyo.Binary)
    model.y = pyo.Var(model.J, within=pyo.Binary)
    model.w = pyo.Var(model.I, model.J, within=pyo.Binary)

    ###  Stage 1

    # Tracking maximum transportation cost from the first stage decisions
    def first_stage_radius_rule(model, i):
        return model.L1 >= sum(c_stage1[i][j] * d_stage1[i] * model.x[i, j] for j in J)

    model.FirstStageRadius = pyo.Constraint(model.I, rule=first_stage_radius_rule)

    # Must open p facilities
    def open_p_rule(model):
        return sum(model.y[j] for j in J) == p

    model.OpenP = pyo.Constraint(rule=open_p_rule)

    # Can only assign a client to a facility if we've opened that facility
    def assign_open_rule(model, i, j):
        return model.x[i, j] <= model.y[j]

    model.AssignOpen = pyo.Constraint(model.I, model.J, rule=assign_open_rule)

    # Every client assigned to exactly one facility
    def assign_one_rule(model, i):
        return sum(model.x[i, j] for j in J) == 1

    model.AssignOne = pyo.Constraint(model.I, rule=assign_one_rule)

    ### Stage 2

    # Tracking maximum transportation cost from the second stage decisions
    def second_stage_radius_rule(model, i):
        return model.L2 >= sum(c2[i][j] * d2[i] * model.w[i, j] for j in J)

    model.SecondStageRadius = pyo.Constraint(model.I, rule=second_stage_radius_rule)

    # Can only assign a client to a facility if we've opened that facility
    def recourse_open_rule(model, i, j):
        return model.w[i, j] <= model.y[j]

    model.RecourseOpen = pyo.Constraint(model.I, model.J, rule=recourse_open_rule)

    # Can only assign a client to a facility if that facility hasn't failed after disruption
    def recourse_available_rule(model, i, j):
        return model.w[i, j] <= 1 - a[j]

    model.RecourseAvailable = pyo.Constraint(
        model.I, model.J, rule=recourse_available_rule
    )

    # Every client assigned to exactly one facility
    def recourse_assign_one_rule(model, i):
        return sum(model.w[i, j] for j in J) == 1

    model.RecourseAssignOne = pyo.Constraint(model.I, rule=recourse_assign_one_rule)

    # Minimize the stage-weighted maximum transportation costs
    # Large multiplicative constant helps ensure that the optimality gap estimates are on
    # the order of magnitude we'd like for downstream analysis
    model.obj = pyo.Objective(
        expr=MULT_CONSTANT * (alpha1 * model.L1 + alpha2 * model.L2),
        sense=pyo.minimize,
    )

    return model


def robust_reliable_pcenter_builder(data, args):
    """
    High-fidelity model:
    a more conservative two-stage reliable p-center over the same realized scenario.

    We keep the same stage structure as LF stochastic, but use a more conservative
    weighting of the second-stage term to represent a higher-fidelity stress-aware model.

    stage 1: x, y
    stage 2: w
    objective: alpha1 * L1 + (alpha2 * "PENALTY") * L2

    """
    I = app_data["I"]
    J = app_data["J"]
    p = app_data["p"]

    alpha1 = app_data["alpha1"]
    alpha2 = app_data["alpha2"]

    c_stage1 = app_data["c_stage1"]
    d_stage1 = app_data["d_stage1"]

    c2 = data["Cost2"]
    d2 = data["Demand2"]
    a = data["Availability"]

    model = pyo.ConcreteModel(data["ID"])

    model.I = pyo.Set(initialize=I)
    model.J = pyo.Set(initialize=J)

    model.L1 = pyo.Var(within=pyo.NonNegativeReals)
    model.L2 = pyo.Var(within=pyo.NonNegativeReals)

    model.x = pyo.Var(model.I, model.J, within=pyo.Binary)
    model.y = pyo.Var(model.J, within=pyo.Binary)
    model.w = pyo.Var(model.I, model.J, within=pyo.Binary)

    ###  Stage 1

    # Tracking maximum transportation cost from the first stage decisions
    def first_stage_radius_rule(model, i):
        return model.L1 >= sum(c_stage1[i][j] * d_stage1[i] * model.x[i, j] for j in J)

    model.FirstStageRadius = pyo.Constraint(model.I, rule=first_stage_radius_rule)

    # Must open p facilities
    def open_p_rule(model):
        return sum(model.y[j] for j in J) == p

    model.OpenP = pyo.Constraint(rule=open_p_rule)

    # Can only assign a client to a facility if we've opened that facility
    def assign_open_rule(model, i, j):
        return model.x[i, j] <= model.y[j]

    model.AssignOpen = pyo.Constraint(model.I, model.J, rule=assign_open_rule)

    # Every client assigned to exactly one facility
    def assign_one_rule(model, i):
        return sum(model.x[i, j] for j in J) == 1

    model.AssignOne = pyo.Constraint(model.I, rule=assign_one_rule)

    ### Stage 2

    # Tracking maximum transportation cost from the second stage decisions
    def second_stage_radius_rule(model, i):
        return model.L2 >= sum(c2[i][j] * d2[i] * model.w[i, j] for j in J)

    model.SecondStageRadius = pyo.Constraint(model.I, rule=second_stage_radius_rule)

    # Can only assign a client to a facility if we've opened that facility
    def recourse_open_rule(model, i, j):
        return model.w[i, j] <= model.y[j]

    model.RecourseOpen = pyo.Constraint(model.I, model.J, rule=recourse_open_rule)

    # Can only assign a client to a facility if that facility hasn't failed after disruption
    def recourse_available_rule(model, i, j):
        return model.w[i, j] <= 1 - a[j]

    model.RecourseAvailable = pyo.Constraint(
        model.I, model.J, rule=recourse_available_rule
    )

    # Every client assigned to exactly one facility
    def recourse_assign_one_rule(model, i):
        return sum(model.w[i, j] for j in J) == 1

    model.RecourseAssignOne = pyo.Constraint(model.I, rule=recourse_assign_one_rule)

    # Slightly more conservative second-stage emphasis than LF stochastic
    # Large multiplicative constant helps ensure that the optimality gap estimates are on
    # the order of magnitude we'd like for downstream analysis
    model.obj = pyo.Objective(
        expr=MULT_CONSTANT * (alpha1 * model.L1 + (alpha2 * 1.5) * model.L2),
        sense=pyo.minimize,
    )

    return model


# ==== CI ADAPTER ==============================================================


class ReliablePCenterCIAdapter(CIProblemAdapter):
    """
    CI adapter supporting:
      - HF = robust reliable p-center
      - LF = either classic p-center or stochastic reliable p-center

    The core Sparow ACV-MRP code expects exactly two active fidelity states:
        - "high"
        - "low"

    Accordingly, this adapter keeps only those two active states.
    The script-specific choice of which low-fidelity model to use is handled
    through the adapter attribute `lf_model_type`, which may be either:
        - "classic"
        - "stochastic"

    Thus:
        active fidelity = "high" means robust reliable p-center
        active fidelity = "low"  means low-fidelity model selected by lf_model_type
    """

    def __init__(
        self,
        model_name,
        scenario_data,
        model_builder,
        app_data=None,
        first_stage_variables=None,
        lf_model_type="classic",
    ):
        self.model_name = model_name
        self.scenario_data = scenario_data
        self.model_builder = model_builder
        self.app_data = {} if app_data is None else dict(app_data)
        self.first_stage_variables = (
            ["x[*,*]", "y[*]"]
            if first_stage_variables is None
            else first_stage_variables
        )

        if lf_model_type not in ("classic", "stochastic"):
            raise ValueError(f"Unknown lf_model_type: {lf_model_type}")

        # Script-specific selector for which concrete model we want to serve as the generic "low" fidelity model
        self.lf_model_type = lf_model_type

        # Core ACV code only expects "high" or "low", default is high
        self._active_fidelity = "high"

    def get_scenario_population(self):
        return self.scenario_data["scenarios"]

    def build_model_data(self, scenarios):
        return {"data": {}, "scenarios": scenarios}

    def build_stochastic_program(self, model_data):
        """
        Build the stochastic program corresponding to the current active fidelity.

        Active fidelity meanings:
            "high" means robust reliable p-center
            "low"  means whichever LF model was selected at adapter construction time via self.lf_model_type
        """
        sp = stochastic_program(first_stage_variables=self.first_stage_variables)
        sp.initialize_application(app_data=self.app_data)

        if self._active_fidelity == "high":
            sp.initialize_model(
                name="HF_RELIABLE_PCENTER",
                model_data=model_data,
                model_builder=robust_reliable_pcenter_builder,
            )

        elif self._active_fidelity == "low":
            if self.lf_model_type == "classic":
                sp.initialize_model(
                    name="LF_CLASSIC_PCENTER",
                    model_data=model_data,
                    model_builder=classic_pcenter_builder,
                )
            elif self.lf_model_type == "stochastic":
                sp.initialize_model(
                    name="LF_STOCHASTIC_PCENTER",
                    model_data=model_data,
                    model_builder=stochastic_reliable_pcenter_builder,
                )
            else:
                raise RuntimeError(f"Unknown lf_model_type: {self.lf_model_type}")

        else:
            raise RuntimeError(f"Unknown active fidelity: {self._active_fidelity}")

        return sp

    def first_stage_variable_order(self):
        order = []
        for i in app_data["I"]:
            for j in app_data["J"]:
                order.append(f"x[{i},{j}]")
        for j in app_data["J"]:
            order.append(f"y[{j}]")
        return order

    def required_scenario_keys(self):
        return ["Demand2", "Cost2", "Availability"]

    def set_active_fidelity(self, fidelity):
        if fidelity not in ("high", "low"):
            raise ValueError(f"Unknown fidelity level: {fidelity}")
        self._active_fidelity = fidelity

    def get_active_fidelity(self):
        return self._active_fidelity

    def get_fidelity_levels(self):
        return ["high", "low"]

    def supports_acv(self):
        return True


# =================================================================
# Core CI code expects exactly one standard factory name
# =================================================================


def get_ci_problem_adapter(model_name="HF", use_integer=False, lf_model_type="classic"):
    """
    Standard factory function expected by the core CI code.

    Notes
    -----
    - model_name controls whether the adapter is initialized in the HF or LF role.
    - lf_model_type controls which concrete model is used whenever the adapter's
      active fidelity is set to "low".
    """
    if model_name == "HF":
        return ReliablePCenterCIAdapter(
            model_name="HF",
            scenario_data=scenario_data_by_model["HF"],
            model_builder=robust_reliable_pcenter_builder,
            app_data=app_data,
            first_stage_variables=["x[*,*]", "y[*]"],
            lf_model_type=lf_model_type,
        )

    if model_name == "LF":
        return ReliablePCenterCIAdapter(
            model_name="LF",
            scenario_data=(
                scenario_data_by_model["LF_classic"]
                if lf_model_type == "classic"
                else scenario_data_by_model["LF_stochastic"]
            ),
            model_builder=(
                classic_pcenter_builder
                if lf_model_type == "classic"
                else stochastic_reliable_pcenter_builder
            ),
            app_data=app_data,
            first_stage_variables=["x[*,*]", "y[*]"],
            lf_model_type=lf_model_type,
        )

    raise ValueError(f"Unknown reliable p-center model_name: {model_name}")


# =================================================================
# Write the scenario data to file for use in the CI tests
# =================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Write the full reliable p-center scenario population to a file "
        "that sparow.ci.cli --scenario-file can read."
    )
    parser.add_argument(
        "--output", required=True, help="Output file path ending in .json or .npy"
    )
    parser.add_argument(
        "--num-data-points",
        type=int,
        default=5,
        help="Number of support points per client for second-stage uncertain demand",
    )
    # This argument doesn't actually affect anything right now...
    # but need to instantiate an adapter in order to get & validate scenario population
    parser.add_argument(
        "--lf-model-type",
        choices=["classic", "stochastic"],
        default="classic",
        help="Select which concrete model should be used whenever ACV-MRP requests low fidelity",
    )
    args = parser.parse_args()

    scenario_object = ReliablePCenterScenarioData(args.num_data_points)
    scenario_data = scenario_object.scenario_generator()

    adapter = ReliablePCenterCIAdapter(
        model_name="HF",
        scenario_data=scenario_data,
        model_builder=robust_reliable_pcenter_builder,
        app_data=app_data,
        first_stage_variables=["x[*,*]", "y[*]"],
        lf_model_type=args.lf_model_type,
    )

    scenarios = adapter.get_scenario_population()
    adapter.validate_scenario_population(scenarios)

    outpath = os.path.abspath(args.output)
    (
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        if os.path.dirname(outpath)
        else None
    )

    if outpath.endswith(".json"):
        with open(outpath, "w") as f:
            json.dump({"scenarios": scenarios}, f, indent=2)
    elif outpath.endswith(".npy"):
        np.save(outpath, {"scenarios": scenarios}, allow_pickle=True)
    else:
        raise ValueError("Output file must end with .json or .npy")

    print(f"Wrote {len(scenarios)} scenarios to: {outpath}")
    print(f"Selected low-fidelity model type: {args.lf_model_type}")
    print(f"Use this with: --scenario-file {outpath}")


if __name__ == "__main__":
    main()
