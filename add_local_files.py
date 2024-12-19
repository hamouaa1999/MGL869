import os
import csv

def find_java_files(directory):
    """
    Recursively find all .java files in the given directory and its subdirectories.
    Returns a list of absolute file paths.
    """
    java_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.abspath(os.path.join(root, file)))
    return java_files

def add_commit_to_csv(csv_file, directory):
    """
    Adds a new row to the CSV file with "Commit String" = "local" and the absolute paths of 
    the Java files found in the specified directory as the "Changed Java Files" column.
    """
    java_files = find_java_files(directory)
    # Ensure we only take the first 42 files if there are more than 42
    java_files = java_files[:134]
    
    # Create the row to add
    commit_string = "local"
    changed_java_files = ",".join(java_files)
    
    new_row = [commit_string, changed_java_files]
    
    # Append the new row to the CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(new_row)
    print(f"Added row: {new_row}")

# Example usage
csv_file = 'path_to_your_csv_filechanged_java_files.csv'  # Replace with the path to your CSV file
directory = '../'  # Replace with the directory you want to search

add_commit_to_csv(csv_file, directory)
