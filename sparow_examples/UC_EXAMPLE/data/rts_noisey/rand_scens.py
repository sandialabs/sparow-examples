import json
import random
import os

# Load the JSON data from the file
with open('nominal_dem.json', 'r') as file:
    data = json.load(file)

# Number of scenarios to create
num_scenarios = 8

# Create a directory to save the scenario files if it doesn't exist
os.makedirs('scenarios', exist_ok=True)

# Generate scenarios
for scenario in range(1, num_scenarios + 1):
    # Create a copy of the original data to modify
    scenario_data = data.copy()  # Copy the entire original data structure
    
    # Modify the demand values
    modified_demand = []
    for demand in data['demand']:
        # Add random noise to the demand value
        noise = random.uniform(-300, 300)  # Adjust the range of noise as needed
        modified_demand.append(demand + noise)
    
    # Update the demand in the copied data
    scenario_data['demand'] = modified_demand
    
    # Save the scenario to a new JSON file
    scenario_filename = f'scenarios/scenario_{scenario}.json'
    with open(scenario_filename, 'w') as scenario_file:
        json.dump(scenario_data, scenario_file, indent=4)

    print(f"Scenario {scenario} saved as {scenario_filename}")
