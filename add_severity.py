
import pandas as pd

# Load the CSV files into DataFrames
file1 = 'cleaned_file_2.0.0.csv'  # File containing "Issue key" and "Priority"
file2 = 'commits_severity_2.0.0.csv'  # File containing "Issue keu" and "bug"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Ensure column names are consistent
df1.rename(columns={"Issue key": "Issue key"}, inplace=True)
df2.rename(columns={"Issue keu": "Issue key"}, inplace=True)

# Merge the two DataFrames on "Issue key", keeping only matching rows
merged_df = df2.merge(df1, on="Issue key", how="inner")

# Update the "bug" column
merged_df["bug"] = merged_df["Priority"]

# Drop the "Priority" column as it's not needed in the final output
merged_df.drop(columns=["Priority"], inplace=True)

# Save the result back to a CSV file
output_file = '2.0.0_with_metrics_and_severity.csv'
merged_df.to_csv(output_file, index=False)

print(f"Updated file saved as {output_file}")

