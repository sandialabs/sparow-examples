import pandas as pd
import matplotlib.pyplot as plt

# Define the lists for PH and EF files
PH_files = [
    'result_jsons/results_HF_PH.json',
    'result_jsons/results_LF_PH.json',
    'result_jsons/results_MF_RANDOM_PH.json',
    'result_jsons/results_MF_SIM_PH.json',
    'result_jsons/results_MF_DISSIM_PH.json'
]

# Updated sample data for demonstration
PH_CPUs = ['1:15:28.794500', '0:12:32.507800', '0:35:48.132000', 
            '0:38:15.480000', '0:55:45.552000']  # Converted to seconds below
PH_objs = [402088.8010, 505611.3605, 459659.3135, 
           461122.4998, 461487.3073]
PH_Iters = [20, 20, 20, 20, 20]  # Set all iterations to 20

EF_files = [
    'result_jsons/results_HF_EF.json',
    'result_jsons/results_LF_EF.json',
    'result_jsons/results_MF_EF.json'
]

# Updated sample data for demonstration
EF_CPUs = ['0:55:54.138000', '0:02:25.368000', '0:18:55.000000']
EF_objs = [401143.1912, 494900.0447, 453275.1612]

# Create DataFrames for PH and EF files
PH_df = pd.DataFrame({
    'PH Files': PH_files,
    'CPU': PH_CPUs,
    'Objective': PH_objs,
    'Iterations': PH_Iters
})

EF_df = pd.DataFrame({
    'EF Files': EF_files,
    'CPU': EF_CPUs,
    'Objective': EF_objs
})

# Get the actual objective value from the EF table (assuming HF_EF is the first entry)
actual_value = EF_df.loc[0, 'Objective']  # This is the objective from HF_EF

# Calculate Percent Error for PH objectives
PH_df['Percent Error'] = ((PH_df['Objective'] - actual_value) / actual_value) * 100

# Display the updated tables
print("PH Files Table with Percent Error:")
print(PH_df)

print("\nEF Files Table:")
print(EF_df)

# Convert CPU time to seconds for plotting
def convert_time_to_seconds(time_str):
    h, m, s = map(float, time_str.split(':'))
    return h * 3600 + m * 60 + s

PH_df['CPU Seconds'] = PH_df['CPU'].apply(convert_time_to_seconds)

# Add HF_EF data for plotting
HF_EF_time = convert_time_to_seconds(EF_CPUs[0])  # Time for HF_EF
HF_EF_percent_error = 0  # Percent error for HF_EF is zero
LF_EF_time = convert_time_to_seconds(EF_CPUs[1])  # Time for LF_EF
LF_EF_percent_error = ((EF_objs[1] - actual_value) / actual_value) * 100
MF_EF_time = convert_time_to_seconds(EF_CPUs[2])  # Time for MF_EF
MF_EF_percent_error = ((EF_objs[2] - actual_value) / actual_value) * 100

cps = list(PH_df['CPU Seconds'])
pes = list(PH_df['Percent Error'])
plot_data = {
    'Method': ['HF_EF', 'HF_PH', 'LF_PH', 'MF_PH_Random', 'MF_PH_Similar', 'MF_PH_Dissimilar', 'LF_EF', 'MF_EF'],
    'CPU Seconds': [HF_EF_time] + [cps[0], cps[1], cps[2], cps[3], cps[4]] + [LF_EF_time, MF_EF_time],
    'Percent Error': [HF_EF_percent_error] + [pes[0], pes[1], pes[2], pes[3], pes[4]] + [LF_EF_percent_error, MF_EF_percent_error]
}

# Create a DataFrame for plotting
plot_df = pd.DataFrame(plot_data)

# Plotting
plt.figure(figsize=(10, 6))

# Define markers and colors for each method
markers = ['x', 'x', '*', 'o', 'o', 'o', '*','o']
colors = ['Black', 'Blue', 'Blue', 'Yellow', 'Green', 'Red', 'Black','Black']

# Loop through each method and plot
for i in range(len(plot_df)):
    plt.scatter(plot_df['CPU Seconds'][i], plot_df['Percent Error'][i], 
                marker=markers[i % len(markers)], 
                color=colors[i], 
                label=plot_df['Method'][i], s=100)

plt.xlabel('CPU Time (seconds)', fontsize=15)
plt.ylabel('Percent Error (%)', fontsize=15)

# Adding legend
plt.axhline(0, color='grey', lw=0.8, ls='--')  # Line at y=0 for reference
plt.grid()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Methods', fontsize=13)
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)

# Show the plot
plt.tight_layout()
plt.savefig('OPF_runs.png')
plt.show()