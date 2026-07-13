import pandas as pd

input_file = "model/data/branch.csv"
output_file = "model/data/branch.csv"
multiplier = 0.1

rating_columns = ["Cont Rating", "LTE Rating", "STE Rating"]

df = pd.read_csv(input_file)

df[rating_columns] = df[rating_columns] * multiplier

df.to_csv(output_file, index=False)

print(f"Updated ratings saved to {output_file}")
