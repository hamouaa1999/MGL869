import pandas as pd

# Load the original CSV file
input_file = "4.0.0.csv"
output_file = "commits_severity_4.0.0.csv"

# Read the CSV file
df = pd.read_csv(input_file)

# Select only the columns "Issue Key" and "Priority"
selected_columns = df[["Issue key", "Priority"]]

# Save the new DataFrame to a new CSV file
selected_columns.to_csv(output_file, index=False)

print(f"New CSV file with 'Issue Key' and 'Priority' saved to {output_file}")
