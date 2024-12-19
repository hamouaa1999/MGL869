import csv
import subprocess
import os

# Input and output file paths
INPUT_CSV = "files_changed_2.0.0_p1.csv"
FINAL_OUTPUT_CSV = "metrics_calculated_2.0.0_p1.csv"
UND_DB_PATH = "/Users/mac/MGL869-laboratoire/laboratoire/hive/UNDERSTAND_DB.und"
UND_APP_PATH = "/Applications/Understand.app/Contents/MacOS/und"

# /Applications/Understand.app/Contents/MacOS/und add /Users/mac/hive/hive/metastore/src/java/org/apache/hadoop/hive/metastore/MetaStoreDirectSql.java -db /Users/mac/MGL869-laboratoire/laboratoire/hive/UNDERSTAND_DB.und
# Commands template
COMMANDS_TEMPLATE = [
    [UND_APP_PATH, 'add', None, '-db', UND_DB_PATH],
    [UND_APP_PATH, 'analyze', '-db', UND_DB_PATH],
    [UND_APP_PATH, 'metrics', '-db', UND_DB_PATH],
    [UND_APP_PATH, 'remove', None, '-db', UND_DB_PATH]
]

def execute_commands(file_path):
    """Executes the Understand commands for the given file."""
    for command in COMMANDS_TEMPLATE:
        # Add the file path to the appropriate commands
        if 'add' in command or 'remove' in command:
            command[2] = file_path
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {command}\n{e}")
            return False
    return True

def extract_metrics(priority, file_path):
    """Extracts metrics from the generated metrics file and adds priority."""
    temp_metrics_file = "metrics.csv"  # Replace this with the actual path Understand uses to output metrics
    extracted_rows = []
    if os.path.exists(temp_metrics_file):
        with open(temp_metrics_file, 'r') as metrics_csv:
            reader = csv.DictReader(metrics_csv)
            for row in reader:
                if row.get("Kind") == "File" and row.get("Name") in file_path:
                    row["Priority"] = priority
                    extracted_rows.append(row)
    else:
        print(f"Metrics file {temp_metrics_file} not found.")
    return extracted_rows

def main():
    # Initialize final output file
    with open(FINAL_OUTPUT_CSV, 'w', newline='') as final_csv:
        writer = None

        # Read input CSV and process each row
        with open(INPUT_CSV, 'r') as input_csv:
            reader = csv.DictReader(input_csv)
            for input_row in reader:
                priority = input_row["Priority"]
                files_changed = input_row["files changed"].split('; ')

                for file_path in files_changed:
                    file_path = file_path.strip()

                    # Execute the commands for the file
                    print(f"Processing file: {file_path}")
                    if execute_commands(file_path):
                        # Extract metrics for the file
                        metrics_rows = extract_metrics(priority, file_path)

                        # Write metrics to the final output CSV
                        for row in metrics_rows:
                            if writer is None:
                                writer = csv.DictWriter(final_csv, fieldnames=row.keys())
                                writer.writeheader()
                            writer.writerow(row)

if __name__ == "__main__":
    main()
