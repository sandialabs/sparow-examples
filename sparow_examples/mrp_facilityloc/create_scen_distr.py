"""
A scenario vector is one realization of all uncertain parameters in the model.
The scenario distribution is the probability distribution over the space of all possible scenarios.

FOR NOW: Assume the components of the scenario vector are independent.

This script takes historical problem data as input and fits a distribution using Gaussian kernel density
estimation (KDE), without assuming a parametric form for the true underlying distribution. This means we fit a separate,
1D gaussian KDE to the historical data of each uncertain parameter.

We can then use these fitted marginal models to draw i.i.d samples of scenario vectors.
Each scenario vector is built by sampling each uncertain parameter independently from its own fitted KDE,
so the approximating joint distribution is the product of the fitted marginals.

When we assign each sampled scenario vector equal probability (1/n), we are treating the n sampled vectors
as an i.i.d. Monte Carlo sample from that approximating joint distribution and using the corresponding
empirical distribution.

Alternatively, the script can assign non-equal scenario weights by approximating probability mass through
bin-integration of the fitted KDEs.

For example, in facility location with AMPL data:
    If there are 4 cities, each scenario vector will have 4 components,
    where the first component is the demand for city 1, the second component is the demand for city 2, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path
from dataclasses import dataclass
from scipy.integrate import quad


@dataclass
class FittedDistributionforUncertainParameter:
    """
    Class to hold the fitted distribution for a one uncertain, scenario-specific model parameter.
    """

    parameter_name: str  # name of the uncertain parameter (e.g., "Demand")
    kde: gaussian_kde  # the fitted gaussian KDE object: used to draw new samples of this uncertain parameter
    nonnegative_flag: bool = False  # whether the parameter has to be nonnegative


class ScenarioDistribution:
    """
    General-purpose scenario distribution approximator.
    FOR NOW: Each uncertain model parameter is fit separately with a 1D Gaussian KDE
             (no modeling of correlations between the parameters yet)

    A single, sampled scenario is a dictionary of the form:
        {
            "ID": "...",
            "<parameter_name_1>": sampled_value,
            "<parameter_name_2>": sampled_value,
            ...
            "Sampled_Vector": [...],  # list of sampled parameter values in the order specified by parameter_names
            "Probability": ... # this is the probability associated with the full scenario vector
        }

    The output can then be used downstream as:
        model_data = {"data": {...}, "scenarios": scenarios}
    """

    def __init__(self, seed=None):

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng(1234)

        # Map parameter name to FittedDistributionforUncertainParameter object
        self.uncertain_parameter_distributions = {}

        # Keeps track of the order in which scenario-vector components (the uncertain parameters) should appear
        self.parameter_names_in_order = []

    def fit_distribution_for_parameter(
        self, parameter_name: str, data, nonnegative_flag=False, save_plot=True
    ):
        """
        Fit a gaussian KDE to the input data for a given uncertain parameter,
        and store the fitted distribution in self.uncertain_parameter_distributions.

        Input data should be a 1d array of historical realizations of the uncertain parameter.
        """

        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise ValueError(
                f"Input data for parameter '{parameter_name}' must be 1D; got shape {data.shape}."
            )

        kde = gaussian_kde(data)
        self.uncertain_parameter_distributions[parameter_name] = (
            FittedDistributionforUncertainParameter(
                parameter_name=parameter_name,
                kde=kde,
                nonnegative_flag=nonnegative_flag,
            )
        )
        if parameter_name not in self.parameter_names_in_order:
            self.parameter_names_in_order.append(parameter_name)

        # Plot the fitted KDE along with a histogram of the orginal input data to visualize the fit
        if save_plot:
            tails_of_distr = 0.02 * (
                max(data) - min(data)
            )  # using 2% (more likely to be feasible)
            x_range = np.linspace(
                min(data) - tails_of_distr, max(data) + tails_of_distr, 1000
            )
            kde_values = kde(x_range)
            plt.figure()
            plt.plot(x_range, kde_values, label="Gaussian KDE")
            plt.hist(
                data, bins=20, density=True, label="Histogram"
            )  # density=True to represent PDF
            plt.xlabel("Data")
            plt.ylabel("Density")
            plt.title(f"Gaussian KDE of {parameter_name}")
            plt.legend()
            plt.savefig(f"histogram_{parameter_name}.png")
            plt.close()

    def sample_one_parameter(self, parameter_name):
        """
        Draw one sample from the fitted distribution of a single uncertain parameter.
        """
        fitted_dist = self.uncertain_parameter_distributions[parameter_name]
        sampled_value = fitted_dist.kde.resample(1, seed=self.rng).item()

        # TODO: is this statistically sound?
        if fitted_dist.nonnegative_flag:
            sampled_value = max(0.0, sampled_value)

        return float(sampled_value)

    def sample_scenarios(
        self, num_scenarios, parameter_names=None, assign_equal_probabilities=True
    ):
        """
        Sample new scenarios from the fitted distributions.

        Output is a list of scenario dictionaries, where each dictionary has the form:
            {
                "ID": "...",
                "<parameter_name_1>": sampled_value,
                "<parameter_name_2>": sampled_value,
                ...
                "Sampled_Vector": [...],  # list of sampled parameter values in the order specified by parameter_names
                "Probability": ... # this is the probability associated with the full scenario vector
            }

        If assign_equal_probabilities is True, each scenario is assigned probability 1/num_scenarios.

        Otherwise, approximate scenario weights are computed by:
            1. binning the sampled values for each parameter,
            2. integrating the fitted KDE over each bin to estimate bin probability mass,
            3. assigning each sampled parameter value the probability mass of its bin,
            4. multiplying across parameters to get a scenario weight,
            5. normalizing the scenario weights to sum to 1.
        """

        if parameter_names is None:
            parameter_names = self.parameter_names_in_order

        for param_name in parameter_names:
            if param_name not in self.uncertain_parameter_distributions:
                raise ValueError(
                    f"No fitted distribution found for parameter '{param_name}'."
                )

        scenarios = []

        # Step 1: sample all scenario vectors first
        for scen_idx in range(num_scenarios):
            scenario_dict = {"ID": f"{scen_idx}"}
            sampled_vector = []

            for parameter_name in parameter_names:
                sampled_value = self.sample_one_parameter(parameter_name)
                scenario_dict[parameter_name] = sampled_value
                sampled_vector.append(sampled_value)

            scenario_dict["Sampled_Vector"] = sampled_vector
            scenarios.append(scenario_dict)

        # Step 2: assign probabilities for each sampled scenario vector
        if assign_equal_probabilities:
            scenario_prob = 1.0 / num_scenarios
            for scenario_dict in scenarios:
                scenario_dict["Probability"] = float(scenario_prob)

        else:
            # We now want to assign approximate probabilities to each sampled value.
            # gaussian KDE gives us a continuous distribution, so we have to estimate the probabilities associated with
            # each sample by integrating the PDFs
            num_bins = 8  # could make this a function argument later

            # For each parameter, compute bin probabilities based on all sampled values for that parameter
            parameter_bin_data = {}

            for parameter_name in parameter_names:
                fitted_dist = self.uncertain_parameter_distributions[parameter_name]

                # This is the array of sampled values for this parameter across all scenarios;
                # we use this to determine the bin edges for the histogram,
                # which we need to do the bin-integration of the KDE to get the bin probabilities
                sampled_values = np.array(
                    [scenario_dict[parameter_name] for scenario_dict in scenarios],
                    dtype=float,
                )

                sample_min = sampled_values.min()
                sample_max = sampled_values.max()
                bin_edges = np.linspace(sample_min, sample_max, num_bins + 1)

                # Calculate the probability mass associated with each bin
                bin_probs = []
                for i in range(num_bins):
                    low, high = bin_edges[i], bin_edges[i + 1]
                    prob, _ = quad(
                        fitted_dist.kde, low, high
                    )  # this step does the integration
                    bin_probs.append(prob)

                # Normalize the bin probabilities so that they sum to 1
                norm_factor = sum(bin_probs)
                if norm_factor > 0.0:
                    bin_probs = [p / norm_factor for p in bin_probs]

                parameter_bin_data[parameter_name] = {
                    "bin_edges": bin_edges,
                    "bin_probs": bin_probs,
                }

            # Assign each scenario vector a probability by multiplying parameter-wise bin masses
            raw_weights = []

            for scenario_dict in scenarios:
                joint_weight = 1.0

                for parameter_name in parameter_names:
                    sampled_value = scenario_dict[parameter_name]
                    bin_edges = parameter_bin_data[parameter_name]["bin_edges"]
                    bin_probs = parameter_bin_data[parameter_name]["bin_probs"]

                    param_prob = probability_by_bin(bin_edges, bin_probs, sampled_value)
                    joint_weight *= param_prob

                raw_weights.append(joint_weight)

            weight_sum = sum(raw_weights)

            if weight_sum <= 0.0:
                raise ValueError(
                    "All bin-integration-based scenario weights are zero; cannot normalize probabilities."
                )

            for scenario_dict, weight in zip(scenarios, raw_weights):
                scenario_dict["Probability"] = float(weight / weight_sum)

        return scenarios


# Helper function that estimates probability of a value sampled from the fitted distribution
# as the probability mass of the bin it falls into
# (i.e. - integrate the KDE over the bin edges to get the probability mass associated with that bin),
# where the bins are defined by the histogram of the original data used to fit the KDE.
def probability_by_bin(bin_edges, bin_probs, value):
    ### THIS FUNCTION IS FROM SANDIA AI -RMA

    # Handle values outside the bin range
    if value < bin_edges[0] or value > bin_edges[-1]:
        return 0.0

    # For value == last edge, assign to last bin
    if value == bin_edges[-1]:
        return bin_probs[-1]

    # Find bin index
    bin_index = np.searchsorted(bin_edges, value, side="right") - 1

    # Safety check
    if bin_index < 0 or bin_index >= len(bin_probs):
        return 0.0

    return bin_probs[bin_index]


if __name__ == "__main__":

    # Original demand data from AMPL example
    ampl_data_original = {
        "San_Antonio_TX": [450, 650, 887],
        "Dallas_TX": [910, 1134, 1456],
        "Jackson_MS": [379, 416, 673],
        "Birmingham_AL": [91, 113, 207],
    }

    # HACK: Since this is a small number of data points per city, use np.linspace to add some more
    # PLEASE NOTE: This is not statistically sound at all, just doing it for toy example
    ampl_data = {}
    for key, val in ampl_data_original.items():

        sorted_vals = np.sort(np.asarray(val, dtype=float))
        min_val, mid_val, max_val = sorted_vals

        lower_segment = np.linspace(min_val, mid_val, 100, endpoint=True)
        upper_segment = np.linspace(mid_val, max_val, 100, endpoint=True)

        # Concatenate, dropping one copy of mid_val so it is not duplicated
        ampl_data[key] = np.concatenate([lower_segment, upper_segment[1:]])

    # Number of scenarios to generate
    n = 100

    # Initialize scenario distribution object
    scen_dist = ScenarioDistribution(seed=42)

    # Fit one KDE per uncertain parameter (here: one per city's demand)
    for parameter_name, historical_values in ampl_data.items():
        scen_dist.fit_distribution_for_parameter(
            parameter_name=parameter_name,
            data=historical_values,
            nonnegative_flag=True,
            save_plot=True,
        )

    # Sample scenarios
    raw_scenarios = scen_dist.sample_scenarios(
        num_scenarios=n,
        parameter_names=list(ampl_data.keys()),
        assign_equal_probabilities=True,  # preffered for SAA
    )

    # Create scenario list in the format model_data expects
    scens_list = []  # list of scenarios to populate
    for idx, scen in enumerate(raw_scenarios):
        scens_list.append(
            {
                "ID": scen["ID"],
                "Demand": scen["Sampled_Vector"],
                "Probability": scen["Probability"],
            }
        )

    # Save scenario list
    output_path = Path(__file__).resolve().parent / "scens_list.npy"
    np.save(output_path, scens_list)

    # Save big-M value based on this batch
    bigM = max(max(scen["Demand"]) for scen in scens_list)
    bigM_path = Path(__file__).resolve().parent / "bigM.txt"
    bigM_path.write_text(str(bigM))

    # Print what was generated
    print("Sampled scenarios with probabilities:")
    for scen in scens_list:
        print(
            f"\n"
            f"Scenario ID: {scen['ID']}, "
            f"Demand: {scen['Demand']}, "
            f"Probability: {scen['Probability']}"
        )
