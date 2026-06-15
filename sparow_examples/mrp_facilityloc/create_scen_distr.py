"""
create scenario distribution from gaussian kernel density estimation (not assuming underlying distribution) based on customer demand values from AMPL example.
each row in the array that's outputted is a scenario, where each index is the demand of a city.
"""

#########################################################

# REPLACE n WITH THE NUMBER OF SAMPLES YOU WANT TO DRAW
n = 4

#########################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import quad

# demand data from AMPL example
ampl_data = {
    "San_Antonio_TX": [450, 650, 887],
    "Dallas_TX": [910, 1134, 1456],
    "Jackson_MS": [379, 416, 673],
    "Birmingham_AL": [91, 113, 207],
}

# create gaussian KDE and evaluate over 1000 points between min-tails and max+tails values of ampl_data:
all_values = np.concatenate(
    list(ampl_data.values())
)  # concatenate values into a single array for KDE
kde = gaussian_kde(all_values)
tails_of_distr = 0.02 * (max(all_values) - min(all_values))  # using 2% (more likely to be feasible)
x_range = np.linspace(
    min(all_values) - tails_of_distr, max(all_values) + tails_of_distr, 1000
)
kde_values = kde(x_range)

# plot gaussian KDE
plt.plot(x_range, kde_values, label="Gaussian KDE")
plt.hist(
    all_values, bins=20, density=True, label="Histogram"
)  # density=True to represent PDF
plt.xlabel("Data")
plt.ylabel("Density")
plt.title("Gaussian KDE of AMPL Data")
plt.legend()
plt.savefig("histogram.png")

# sample from gaussian KDE
samples = kde.resample(4 * n)  # sample n times (each scenario has dim 4)
samples = np.clip(
    samples, a_min=0, a_max=None
)  # replace negatives with 0... TODO: is this statistically sound???

# gaussian KDE gives us a continuous distribution so we have to estimate the probabilities associated with each sample by integrating the PDF
num_bins = 8    # number of bins in the histogram; if any larger than 8, some bins will be empty
bin_edges = np.linspace(samples.min(), samples.max(), num_bins + 1)

# saving the maximum sample to use in the big-M model 
with open("bigM.txt", "w") as file:
    file.write(str(samples.max()))

# function that estimates probability of a demand value as the density of the bin it falls into
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

# calculate the density associated with each bin
bin_probs = []
for i in range(num_bins):
    low, high = bin_edges[i], bin_edges[i + 1]
    prob, _ = quad(kde, low, high)  # this step does the integration
    bin_probs.append(prob)
norm_factor = sum(bin_probs)

# estimate the probability associated with each demand value
sample_probs = []
for sample in samples:
    for dem_in_one_city in sample:
        sample_probs.append(probability_by_bin(bin_edges, bin_probs, dem_in_one_city))

# reshape the samples so that each row corresponds to a scenario
samples = samples.reshape(n, 4)  # reshape array so each row corresponds to a scenario
# turn sample_probs into a numpy array so that it can similarly be reshaped
sample_probs = np.array(sample_probs)
sampe_probs = sample_probs.reshape(n, 4)

# create scenario list in the format model_data expects
scens_list = []  # list of scenarios to populate
for idx, scen in enumerate(samples):
    scens_list.append(
        {"ID": f"{idx}", "Demand": scen.tolist(), "Probability": np.prod(sample_probs[idx])}
    )
# normalize HF scenario probabilities
norm_term = sum(scens_list[s_idx]["Probability"] for s_idx in range(len(scens_list)))
for s_idx in range(len(scens_list)):
    scens_list[s_idx]["Probability"] /= norm_term

np.save("scens_list.npy", scens_list)
