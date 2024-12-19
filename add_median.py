import pandas as pd

# Load the CSV file into a DataFrame
file_path = "metrics_calculated_4.0.0_p1.csv"  # Replace with the actual file path
df = pd.read_csv(file_path, na_values=["", " ", "NA"])  # Treat empty strings and "NA" as NaN

# Fill empty cells with the median of their respective columns
for column in df.columns:
    if df[column].dtype in ['float64', 'int64']:  # Ensure column is numeric
        if df[column].isna().sum() > 0:  # Check if there are any NaN values
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)

# Save the updated DataFrame to a new CSV file (optional)
output_file_path = "filled_median_file_4.0.0_p1.csv"
df.to_csv(output_file_path, index=False)

print("Empty cells have been filled with the median values.")

import pandas as pd

# Load the CSV file into a DataFrame
file_path = "filled_median_file_4.0.0_p1.csv"  # Replace with the actual file path
df = pd.read_csv(file_path, na_values=["", " ", "NA"])  # Treat empty strings and "NA" as NaN

# Identify and remove columns where all rows are NaN
columns_before = df.shape[1]
df = df.dropna(axis=1, how="all")  # Drop columns where all values are NaN
columns_after = df.shape[1]

# Calculate the number of columns removed
columns_removed = columns_before - columns_after

# Save the updated DataFrame to a new CSV file (optional)
output_file_path = "cleaned_file_4.0.0_p1.csv"
df.to_csv(output_file_path, index=False)

# Report the results
print(f"{columns_removed} columns were removed because all their rows were NaN.")