import csv
import subprocess
import os

# Input and output file paths
INPUT_CSV = "files_changed_4.0.0_p1.csv"
FINAL_OUTPUT_CSV = "metrics_calculated_4.0.0_p1.csv"
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

# Function to run git log command and count the number of commits
def count_commits(file_path, release_range="HEAD"):
    # Command to count commits that modified the given file in the specified release range
    command = f"git log --oneline {release_range} -- {file_path} | wc -l"
    try:
        # Execute the command and get the result
        result = subprocess.check_output(command, shell=True)
        return int(result.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error while executing command: {e}")
        return 0

# Function to count commits with the keyword "fix" in their messages
def count_fix_commits(file_path, release_range="HEAD"):
    command = f"git log --oneline --grep='fix' {release_range} -- {file_path} | wc -l"
    try:
        result = subprocess.check_output(command, shell=True)
        return int(result.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error while executing command: {e}")
        return 0

# Function to calculate the number of added lines between two releases
def count_added_lines(file_path, start_commit, end_commit):
    # Git command to count added lines
    command = f"git diff --shortstat {start_commit} {end_commit} -- {file_path} | awk '/insertions/ {{print $4}}'"
    try:
        result = subprocess.check_output(command, shell=True)
        # Ensure the result is not empty before converting to int
        return int(result.strip()) if result.strip() else 0
    except subprocess.CalledProcessError as e:
        print(f"Error while executing command: {e}")
        return 0

# Function to calculate the number of deleted lines between two releases
def count_deleted_lines(file_path, start_commit, end_commit):
    # Git command to count deleted lines
    command = f"git diff --shortstat {start_commit} {end_commit} -- {file_path} | awk '/deletions/ {{print $6}}'"
    try:
        result = subprocess.check_output(command, shell=True)
        # Ensure the result is not empty before converting to int
        return int(result.strip()) if result.strip() else 0
    except subprocess.CalledProcessError as e:
        print(f"Error while executing command: {e}")
        return 0


import csv
import subprocess
from datetime import datetime

# Helper function to run a shell command and return the output
def run_command(command):
    try:
        result = subprocess.check_output(command, shell=True, text=True)
        return result.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error while executing command: {e}")
        return ""

# Function to count developers who changed the file in a specific version
def count_developers(file_path, release_range="HEAD"):
    command = f"git log --format='%ae' {release_range} -- {file_path} | sort | uniq | wc -l"
    result = run_command(command)
    return int(result) if result else 0

# Function to count developers who changed the file in the version and all prior versions
def count_developers_all_versions(file_path, release_range="HEAD"):
    command = f"git log --format='%ae' {release_range} -- {file_path} | sort | uniq | wc -l"
    result = run_command(command)
    return int(result) if result else 0

# Function to calculate average time between commits in a version
def average_time_between_commits(file_path, release_range="HEAD"):
    command = f"git log --format='%ct' {release_range} -- {file_path}"
    result = run_command(command)
    timestamps = list(map(int, result.splitlines())) if result else []
    if len(timestamps) < 2:
        return 0  # Not enough commits to calculate average time
    intervals = [timestamps[i] - timestamps[i + 1] for i in range(len(timestamps) - 1)]
    average_interval = sum(intervals) / len(intervals)
    return average_interval / 3600  # Convert seconds to hours

# Function to calculate average time between commits for all versions up to V
def average_time_between_commits_all_versions(file_path, release_range="HEAD"):
    return average_time_between_commits(file_path, release_range)

def average_expertise(file_path, release_range="HEAD"):
    command = f"git log --format='%ae' {release_range} -- {file_path} | sort | uniq"
    developers = run_command(command).splitlines()
    total_expertise = 0

    # Extract end commit for expertise calculation
    end_commit = release_range.split('..')[1] if '..' in release_range else None

    for dev in developers:
        if end_commit:
            expertise_command = f"git log --format='%h' --author='{dev}' --before='{end_commit}' | wc -l"
        else:
            expertise_command = f"git log --format='%h' --author='{dev}' | wc -l"
        expertise = run_command(expertise_command)
        total_expertise += int(expertise) if expertise else 0

    return total_expertise / len(developers) if developers else 0


def min_expertise(file_path, release_range="HEAD"):
    command = f"git log --format='%ae' {release_range} -- {file_path} | sort | uniq"
    developers = run_command(command).splitlines()
    expertise_values = []

    # Extract end commit for expertise calculation
    end_commit = release_range.split('..')[1] if '..' in release_range else None

    for dev in developers:
        if end_commit:
            expertise_command = f"git log --format='%h' --author='{dev}' --before='{end_commit}' | wc -l"
        else:
            expertise_command = f"git log --format='%h' --author='{dev}' | wc -l"
        expertise = run_command(expertise_command)
        if expertise:
            expertise_values.append(int(expertise))

    # Use a distinct variable for the minimum value
    min_expertise_value = min(expertise_values) if expertise_values else 0
    return min_expertise_value



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
                    
                    commit_count = count_commits(file_path, "release-1.2.1..rel/release-4.0.0")
                    fix_commit_count = count_fix_commits(file_path, "rel/release-4.0.0")
                    added_lines = count_added_lines(file_path, "release-1.2.1", "rel/release-4.0.0")
                    deleted_lines = count_deleted_lines(file_path, "release-1.2.1", "rel/release-4.0.0")
                    commit_count_before_release = count_commits(file_path, "rel/release-4.0.0")

                    developers_in_version = count_developers(file_path, "rel/release-4.0.0")
                    developers_all_versions = count_developers_all_versions(file_path, "release-1.2.1..rel/release-4.0.0")
                    avg_time_between_commits = average_time_between_commits(file_path, "rel/release-4.0.0")
                    avg_time_between_commits_all_versions = average_time_between_commits_all_versions(file_path, "release-1.2.1..rel/release-4.0.0")
                    avg_expertise = average_expertise(file_path, "rel/release-4.0.0")
                    min_expertise_value = min_expertise(file_path, "rel/release-4.0.0")

                    # Add new calculated columns to the row
                    row['CommitCount'] = commit_count
                    row['FixCommitCount'] = fix_commit_count
                    row['AddedLines'] = added_lines
                    row['DeletedLines'] = deleted_lines
                    row['CommitCountBeforeRelease'] = commit_count_before_release
                    row['DevelopersInVersion'] = developers_in_version
                    row['DevelopersAllVersions'] = developers_all_versions
                    row['AvgTimeBetweenCommits'] = avg_time_between_commits
                    row['AvgTimeBetweenCommitsAllVersions'] = avg_time_between_commits_all_versions
                    row['AvgExpertise'] = avg_expertise
                    row['MinExpertise'] = min_expertise_value

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
                files_changed = input_row["files changed"].split(';')

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
