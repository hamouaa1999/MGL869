import csv

def remove_columns(input_file, output_file, columns_to_remove):
    """
    Removes specified columns from the input CSV file and writes the result to the output CSV file.
    
    :param input_file: Path to the input CSV file.
    :param output_file: Path to the output CSV file where the modified data will be saved.
    :param columns_to_remove: List of columns to remove by name.
    """
    with open(input_file, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        # Determine which columns to keep by removing the ones to remove
        columns_to_keep = [column for column in fieldnames if column not in columns_to_remove]

        # Open the output CSV file and write the modified data
        with open(output_file, mode='w', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=columns_to_keep)
            writer.writeheader()

            for row in reader:
                # Write only the columns that are not removed
                row_to_write = {key: value for key, value in row.items() if key in columns_to_keep}
                writer.writerow(row_to_write)

    print(f"Columns {columns_to_remove} have been removed and saved to {output_file}.")

# Example usage
input_file = 'all_versions.csv'  # Replace with the path to your input CSV file
output_file = 'dataset.csv'  # Replace with the desired output file path
columns_to_remove = ['Kind', 'Name']

remove_columns(input_file, output_file, columns_to_remove)
