import os
import sys
import pandas as pd
from DFCreation import extract_peaks

def process_files(data_folder, valid_csv_path, invalid_csv_path):
    # Initialize results list and counters
    csvFileResults = []
    invalid_files_list = []

    valid_files = 0
    invalid_files = 0
    
    try:
        total_files = len(os.listdir(data_folder))
    except FileNotFoundError:
        print(f"Error: The specified data folder '{data_folder}' does not exist.")
        sys.exit(1)

    # Loop through files in the directory
    for file in os.listdir(data_folder):
        full_file_path = os.path.join(data_folder, file)
        
        file_result = extract_peaks(full_file_path)
        
        # Check if the result is a valid dictionary
        if isinstance(file_result, dict):
            csvFileResults.append(file_result)
            valid_files += 1  
        else:
            print(f"Skipping file {file}: Invalid result returned.")
            invalid_files += 1 
            invalid_files_list.append({"file_name": file})  
        
        # Progress tracker
        total_files_processed = valid_files + invalid_files
        print(f"Processed {total_files_processed}/{total_files} files...")

    # Print percentage of valid files
    if total_files > 0:
        valid_percentage = (valid_files / total_files) * 100
    else:
        valid_percentage = 0
    print(f"Valid files: {valid_files}/{total_files} ({valid_percentage:.2f}%)")

    # Save valid results to CSV
    if csvFileResults:
        ClusteringDF = pd.DataFrame(csvFileResults)
        ClusteringDF.to_csv(valid_csv_path, index=False)
        print(f"Valid results saved successfully to '{valid_csv_path}'!")
    else:
        print("No valid data to save. Check the logs.")

    # Save invalid files to a separate CSV
    if invalid_files_list:
        InvalidFilesDF = pd.DataFrame(invalid_files_list)
        os.makedirs(os.path.dirname(invalid_csv_path), exist_ok=True)
        InvalidFilesDF.to_csv(invalid_csv_path, index=False)
        print(f"Invalid files saved successfully to '{invalid_csv_path}'!")
    else:
        print("No invalid files to save.")

    print(f"Processing complete. {valid_percentage:.2f}% of files were valid.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python process_files.py <data_folder> <valid_csv_path> <invalid_csv_path>")
        sys.exit(1)

    data_folder = sys.argv[1]
    valid_csv_path = sys.argv[2]
    invalid_csv_path = sys.argv[3]

    process_files(data_folder, valid_csv_path, invalid_csv_path)
