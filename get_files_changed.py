import pandas as pd
import subprocess
import os

# Paths
input_csv = "commits_severity_4.0.0.csv"  # Generated file with "Issue key" and "Priority"
output_csv = "files_changed_4.0.0_p1.csv"    # New CSV file to generate
repo_path = "../"           # Path to your Git repository

# Load the generated CSV file
df = pd.read_csv(input_csv)

# Function to get changed files for a given Issue key
def get_changed_files(issue_key):
    try:
        # Use git log to find the commit hash for the Issue key
        result = subprocess.run(
            ["git", "-C", repo_path, "log", "--pretty=format:%H", "--grep", f"^{issue_key}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        commit_hashes = result.stdout.splitlines()
        
        if not commit_hashes:
            return None  # No matching commit, return None

        # Collect all Java files changed in those commits
        files = set()
        for commit_hash in commit_hashes:
            diff_result = subprocess.run(
                ["git", "-C", repo_path, "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            files.update(diff_result.stdout.splitlines())

        # Filter for .java files only
        java_files = [file for file in files if file.endswith(".java")]

        # Convert to absolute paths
        absolute_java_files = [os.path.abspath(os.path.join(repo_path, file)) for file in java_files]
        return "; ".join(absolute_java_files) if absolute_java_files else None

    except Exception as e:
        return None  # Handle errors gracefully and exclude row

# Add a new column with files changed
df["files changed"] = df["Issue key"].apply(get_changed_files)

# Filter out rows where "files changed" is None
filtered_df = df.dropna(subset=["files changed"])

# Save the updated DataFrame to a new CSV file
filtered_df.to_csv(output_csv, index=False)

print(f"New CSV file with 'files changed' saved to {output_csv}")

