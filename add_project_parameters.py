import csv
import subprocess

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

# Function to calculate average expertise of developers who changed the file
def average_expertise(file_path, release_range="HEAD"):
    command = f"git log --format='%ae' {release_range} -- {file_path} | sort | uniq"
    developers = run_command(command).splitlines()
    total_expertise = 0
    for dev in developers:
        expertise_command = f"git log --format='%h' --author='{dev}' --before='{release_range.split('..')[1]}' | wc -l"
        expertise = run_command(expertise_command)
        total_expertise += int(expertise) if expertise else 0
    return total_expertise / len(developers) if developers else 0

# Function to calculate minimum expertise of developers who changed the file
def min_expertise(file_path, release_range="HEAD"):
    command = f"git log --format='%ae' {release_range} -- {file_path} | sort | uniq"
    developers = run_command(command).splitlines()
    expertise_values = []
    for dev in developers:
        expertise_command = f"git log --format='%h' --author='{dev}' --before='{release_range.split('..')[1]}' | wc -l"
        expertise = run_command(expertise_command)
        if expertise:
            expertise_values.append(int(expertise))
    # Return the minimum expertise if available; otherwise, return 0
    return min(expertise_values) if expertise_values else 0


# Read the input CSV file and process each row
def process_csv(input_file, output_file):
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_file, mode='w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        fieldnames += [
            'CommitCount', 'FixCommitCount', 'AddedLines', 'DeletedLines', 'CommitCountBeforeRelease',
            'DevelopersInVersion', 'DevelopersAllVersions', 'AvgTimeBetweenCommits',
            'AvgTimeBetweenCommitsAllVersions', 'AvgExpertise', 'MinExpertise'
        ]

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            file_path = row['FILE_PATH']

            # Calculate the required values
            commit_count = count_commits(file_path, "release-1.2.1..release-2.0.0")
            fix_commit_count = count_fix_commits(file_path, "release-2.0.0")
            added_lines = count_added_lines(file_path, "release-1.2.1", "release-2.0.0")
            deleted_lines = count_deleted_lines(file_path, "release-1.2.1", "release-2.0.0")
            commit_count_before_release = count_commits(file_path, "release-2.0.0")

            developers_in_version = count_developers(file_path, "release-2.0.0")
            developers_all_versions = count_developers_all_versions(file_path, "release-1.2.1..release-2.0.0")
            avg_time_between_commits = average_time_between_commits(file_path, "release-2.0.0")
            avg_time_between_commits_all_versions = average_time_between_commits_all_versions(file_path, "release-1.2.1..release-2.0.0")
            avg_expertise = average_expertise(file_path, "release-2.0.0")
            min_expertise = min_expertise(file_path, "release-2.0.0")

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
            row['MinExpertise'] = min_expertise

            # Write the updated row to the new CSV file
            writer.writerow(row)

# Main function to run the program
def main():
    input_file = 'cleaned_file_2.0.0_p1.csv'  # The input CSV file
    output_file = 'project_metrics_2.0.0.csv'  # The output CSV file with added columns

    # Process the CSV file
    process_csv(input_file, output_file)
    print(f"CSV file processed and saved to {output_file}")

# Run the main function
if __name__ == "__main__":
    main()
